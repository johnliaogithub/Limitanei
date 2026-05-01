"""Cascaded PID controller + rotor mixer for an X-frame quadrotor.

Pipeline
--------
   setpoint (pos_des, yaw_des)
        |
        v   [Position PID]      world-frame desired acceleration
        |   ---------------->   = Kp*(pos_des-pos) + Kd*(-vel) + Ki*int + g*z_hat
        v
   desired tilt (roll_des, pitch_des) + total thrust magnitude
        |
        v   [Attitude PID]      body-frame torque tau = (tau_x, tau_y, tau_z)
        |   ---------------->   = Kp*att_err - Kd*omega_body  (scaled by inertia)
        v
   (T_total, tau_x, tau_y, tau_z)
        |
        v   [Mixer]             per-rotor thrust commands
        |   ---------------->   T_i = M^-1 * [T, tau_x, tau_y, tau_z]
        v
   [T1, T2, T3, T4] -> MuJoCo actuators
"""
import numpy as np

from config import DroneParams, ControlParams


# ---------- small math helpers ----------

def quat_to_euler(q):
    """Convert MuJoCo quaternion [w, x, y, z] to (roll, pitch, yaw) in radians.

    Uses the Z-Y-X (yaw-pitch-roll) intrinsic convention, which is the
    standard "aerospace" Euler set.
    """
    w, x, y, z = q
    # roll  (rotation about body X)
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)
    # pitch (rotation about body Y)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    # yaw   (rotation about body Z)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)
    return roll, pitch, yaw


def wrap_pi(a):
    """Wrap an angle to (-pi, pi]. Stops yaw error from going the long way around."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# ---------- main controller ----------

class CascadedController:
    """Cascaded position + attitude PID with rotor mixing."""

    def __init__(self, drone: DroneParams, ctrl: ControlParams,
                 total_mass: float = None, inertia: tuple = None):
        """`total_mass` and `inertia` override drone defaults to account for
        a payload (e.g. a mounted weapon + ammunition). If omitted, the bare
        airframe values from `drone` are used.
        """
        self.d = drone
        self.c = ctrl
        self.m = total_mass if total_mass is not None else drone.mass
        if inertia is None:
            self.Ix, self.Iy, self.Iz = drone.Ixx, drone.Iyy, drone.Izz
        else:
            self.Ix, self.Iy, self.Iz = inertia
        self.pos_int = np.zeros(3)
        self.att_int = np.zeros(3)

        # ---------- Build the mixer matrix ----------
        # Rotor positions in the body frame (X-config).
        # x = forward, y = left, z = up.
        # L is the offset along each axis: arm_length is the diagonal,
        # so each motor sits at (+/- L, +/- L) with L = arm/sqrt(2).
        L = drone.arm_length / np.sqrt(2)
        cq = drone.k_q_over_k_t  # converts thrust force to drag yaw-torque

        # Rotor convention (looking down at the drone, +x = forward):
        #   r1 = front-right (+L, -L), spin direction s = +1 (CCW from above)
        #   r2 = front-left  (+L, +L), spin direction s = -1 (CW)
        #   r3 = back-left   (-L, +L), spin direction s = +1 (CCW)
        #   r4 = back-right  (-L, -L), spin direction s = -1 (CW)
        # Diagonal pairs spin the same way so net yaw torque cancels at hover.
        #
        # A thrust T_i applied at (x_i, y_i, 0) along +body_z gives:
        #   force  on body = (0, 0, T_i)
        #   moment on body = (y_i * T_i, -x_i * T_i, 0)   (r x F)
        #   plus pure yaw drag torque  s_i * cq * T_i  about +z.
        #
        # Stacking [T_total, tau_x, tau_y, tau_z]^T = M * [T1, T2, T3, T4]^T :
        M = np.array([
            [ 1.0,   1.0,   1.0,   1.0],   # total thrust
            [ -L,     L,     L,    -L ],   # roll  torque  (sum of y_i * T_i)
            [ -L,    -L,     L,     L ],   # pitch torque  (sum of -x_i * T_i)
            [  cq,   -cq,    cq,   -cq],   # yaw   torque  (sum of s_i * cq * T_i)
        ])
        self.M_inv = np.linalg.inv(M)

    def reset(self):
        """Clear integrator state (call after a teleport / 'R' reset)."""
        self.pos_int[:] = 0.0
        self.att_int[:] = 0.0

    def update(self, state, setpoint, dt):
        """Run one control step. Returns thrust commands [T1..T4] in newtons."""
        pos = state['pos']
        vel = state['vel']                     # world frame
        omega = state['omega_body']            # body frame angular velocity
        roll, pitch, yaw = quat_to_euler(state['quat'])

        pos_des = setpoint['pos_des']
        yaw_des = setpoint['yaw_des']

        # ---------- OUTER LOOP: position controller ----------
        pos_err = pos_des - pos
        vel_err = -vel                         # we want to come to rest at the setpoint
        self.pos_int += pos_err * dt
        self.pos_int = np.clip(self.pos_int, -self.c.integral_clip, self.c.integral_clip)

        kp = np.array(self.c.kp_pos)
        ki = np.array(self.c.ki_pos)
        kd = np.array(self.c.kd_pos)

        # Desired acceleration in the WORLD frame, including the gravity-cancel term.
        # The +g*z_hat means: if pos_err and vel are zero, we still need to push up
        # with enough thrust to fight gravity.
        acc_des = (
            kp * pos_err
            + ki * self.pos_int
            + kd * vel_err
            + np.array([0.0, 0.0, self.d.gravity])
        )

        # The drone can only push along its own +z axis. Decompose acc_des into:
        #   (a) total thrust magnitude  T_total
        #   (b) the tilt (roll_des, pitch_des) needed to point body-z that way
        # We rotate the desired horizontal accel into the yaw-aligned frame so
        # roll/pitch are independent of which way the drone is facing.
        cy, sy = np.cos(yaw), np.sin(yaw)
        ax_b =  cy * acc_des[0] + sy * acc_des[1]   # forward accel in yaw frame
        ay_b = -sy * acc_des[0] + cy * acc_des[1]   # left    accel in yaw frame
        az = acc_des[2]

        max_tilt = np.radians(self.c.max_tilt_deg)
        # Small-angle relations:
        #   pitch (about +y) tilts thrust toward +x  =>  pitch_des = atan2(ax, az)
        #   roll  (about +x) tilts thrust toward -y  =>  roll_des  = atan2(-ay, az)
        pitch_des = np.clip(np.arctan2(ax_b, az), -max_tilt, max_tilt)
        roll_des  = np.clip(np.arctan2(-ay_b, az), -max_tilt, max_tilt)

        # Total thrust: project a_des onto current body-z (i.e. divide out current tilt).
        cos_rp = max(np.cos(roll) * np.cos(pitch), 0.5)   # avoid div-by-zero on flips
        thrust_total = self.m * az / cos_rp

        # ---------- INNER LOOP: attitude controller ----------
        att_err = np.array([
            wrap_pi(roll_des  - roll),
            wrap_pi(pitch_des - pitch),
            wrap_pi(yaw_des   - yaw),
        ])
        self.att_int += att_err * dt
        self.att_int = np.clip(self.att_int, -self.c.integral_clip, self.c.integral_clip)

        kpa = np.array(self.c.kp_att)
        kia = np.array(self.c.ki_att)
        kda = np.array(self.c.kd_att)

        # PD on angle, with the D-term taken from body angular velocity directly
        # (avoids differentiating a noisy angle signal).
        ang_acc = kpa * att_err + kia * self.att_int - kda * omega

        # Convert desired angular acceleration to a torque using tau = I * alpha.
        # Treating the inertia tensor as diagonal (Ix, Iy, Iz).
        tau = np.array([
            self.Ix * ang_acc[0],
            self.Iy * ang_acc[1],
            self.Iz * ang_acc[2],
        ])

        # ---------- MIXER ----------
        wrench = np.array([thrust_total, tau[0], tau[1], tau[2]])
        T_per_rotor = self.M_inv @ wrench
        # Saturate to physical motor limits. Real motors can't push negative thrust.
        return np.clip(T_per_rotor, 0.0, self.d.max_thrust_per_rotor)
