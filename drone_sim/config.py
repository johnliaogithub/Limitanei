"""Adjustable constants for the drone simulation.

Edit any field below and re-run main.py — every other module reads from
these dataclasses, so there are no constants hidden elsewhere.

Defaults model a ~2.5 kg heavy-lift X-quadrotor (similar to a 12-15 inch
prop frame, e.g. Tarot 650 / S550 class) sized so it can actually carry
a battle rifle or light machine gun and absorb the recoil.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class DroneParams:
    # ---------- Physical body (DRONE ALONE, no payload) ----------
    mass: float = 2.5                    # kg, frame + battery + motors + flight controller
    arm_length: float = 0.35             # m, motor distance from center
    Ixx: float = 0.050                   # kg*m^2, roll inertia
    Iyy: float = 0.050                   # kg*m^2, pitch inertia
    Izz: float = 0.090                   # kg*m^2, yaw inertia (always largest for a flat frame)

    # ---------- Rotor / propeller (~15-inch carbon prop, BLDC motor) ----------
    # Thrust  T = k_thrust * omega^2     [N]      (omega in rad/s)
    # Torque  Q = k_torque * omega^2     [N*m]    (drag torque on body, opposes spin)
    k_thrust: float = 2.5e-4
    k_torque: float = 5.0e-6
    omega_max: float = 600.0             # rad/s (~5700 RPM, realistic for a 15-in prop)
    motor_tau: float = 0.05              # s, motor first-order time constant

    # ---------- Environment ----------
    gravity: float = 9.81                # m/s^2

    # ---------- Derived helpers ----------
    @property
    def hover_omega(self) -> float:
        return float(np.sqrt(self.mass * self.gravity / (4 * self.k_thrust)))

    @property
    def max_thrust_per_rotor(self) -> float:
        return self.k_thrust * self.omega_max ** 2

    @property
    def k_q_over_k_t(self) -> float:
        return self.k_torque / self.k_thrust


@dataclass
class ControlParams:
    # ---------- Position PID gains (x, y, z, world frame) ----------
    # Horizontal integral terms are non-zero so that a constant disturbance
    # (e.g. sustained recoil) is driven out instead of producing a steady
    # offset error.
    kp_pos: tuple = (3.0, 3.0, 5.0)
    ki_pos: tuple = (1.0, 1.0, 1.0)
    kd_pos: tuple = (2.5, 2.5, 3.0)

    # ---------- Attitude PID gains (roll, pitch, yaw, body frame) ----------
    # Outputs are scaled by inertia inside the controller, so these are in
    # angular-acceleration units (rad/s^2 per rad of error). Closed-loop
    # bandwidth is ~ sqrt(kp), independent of inertia, so big drone uses the
    # same gains as a small one - just be aware that motor lag and saturation
    # impose a practical ceiling.
    kp_att: tuple = (200.0, 200.0, 60.0)
    ki_att: tuple = (0.0, 0.0, 0.0)
    kd_att: tuple = (28.0, 28.0, 10.0)

    # ---------- Limits ----------
    max_tilt_deg: float = 45.0           # caps desired roll/pitch from outer loop
    max_yaw_rate_dps: float = 90.0       # max yaw rate command in keyboard mode (deg/s)
    max_climb_rate: float = 3.0          # m/s, used to drive setpoint in keyboard mode
    max_horiz_speed: float = 4.0         # m/s, used to drive setpoint in keyboard mode
    integral_clip: float = 5.0           # anti-windup clamp


@dataclass
class SimParams:
    timestep: float = 0.002              # s, MuJoCo physics step (500 Hz)
    control_hz: float = 250.0            # Hz, controller rate (every other physics step)
    realtime: bool = True                # if True, sleep so 1 sim-second = 1 wall-second

    # Initial state of the drone
    init_pos: tuple = (0.0, 0.0, 1.0)    # m
    init_yaw: float = 0.0                # rad


@dataclass
class GunParams:
    """Gun configuration applied at startup (the model is selected from
    gun.py's catalog by name)."""
    selected: str = "m4_carbine"         # key into gun.GUNS — change at the CLI with --gun
    auto_fire_in_auto_mode: bool = True  # if True, drone fires throughout the auto demo


@dataclass
class DisturbanceParams:
    """All stochastic disturbances. Set sigmas to 0 to disable any of them."""
    seed: int = 0                        # RNG seed; 0 = use system entropy

    # Wind: world-frame mean velocity + turbulent gusts (Ornstein-Uhlenbeck).
    wind_mean_mps: tuple = (0.0, 0.0, 0.0)
    wind_gust_sigma_mps: float = 0.0     # std dev of each component
    wind_gust_tau_s: float = 2.0         # turbulence correlation time
    drone_drag_Cd_A: float = 0.025       # Cd*A product for the airframe (m^2)

    # Per-shot recoil noise.
    # impulse_sigma  = 0.02 means 2% std-dev variability per shot (typical milspec ammo)
    # angular_sigma  = 0.5 deg makes the bullet line wobble slightly per shot
    recoil_impulse_sigma: float = 0.0
    recoil_angular_sigma_deg: float = 0.0

    # Casing ejection variability (only active with --projectiles).
    casing_eject_speed_sigma: float = 0.4
    casing_eject_angle_sigma_deg: float = 8.0


@dataclass
class ProjectileParams:
    """Optional bullet + casing simulation. Off by default."""
    enabled: bool = False
    casing_pool_size: int = 60
    casing_lifetime_s: float = 5.0
    bullet_pool_size: int = 64
    bullet_lifetime_s: float = 1.5
    prop_collision_warn: bool = True     # print a warning when a casing hits a prop


# Singletons — import these from anywhere:  from config import DRONE
DRONE = DroneParams()
CTRL = ControlParams()
SIM = SimParams()
GUN = GunParams()
DIST = DisturbanceParams()
PROJ = ProjectileParams()
