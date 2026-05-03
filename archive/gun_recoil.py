"""Gun recoil physics simulation using PyBullet.

Modeled weapon: 4.0 kg battle rifle in 7.62x51 NATO (.308-class) — comparable
to an FN FAL / M14 / SCAR-H. 10 g bullet at 850 m/s with a gas-ejecta factor
of 1.6 to account for propellant gases.

The rifle is a free rigid body held by two virtual spring-dampers (shoulder
and grip). Firing applies the recoil impulse over the brief gas-pressure
window (~3 ms) instead of a single instant kick, so the gun visibly travels
rearward, the muzzle pitches up, then the springs pull it back to rest.

Controls:
    SPACE     hold for full-auto, tap for single shot
    R         reset gun pose
    ESC       quit
"""

import math
import random
import time

import numpy as np
import pybullet as p
import pybullet_data


GUN_MASS = 4.0           # kg — battle rifle (FAL/M14/SCAR-H class)
BULLET_MASS = 0.010      # kg — 7.62x51 NATO (.308 Win) bullet
MUZZLE_VELOCITY = 850.0  # m/s
EJECTA_FACTOR = 1.6      # gas/powder roughly adds 60% more momentum
RECOIL_PULSE = 0.003     # s, gas pressure window

CYCLIC_RPM = 650.0       # full-auto rate of fire (FAL ~700, M14 ~750, AK ~600)
FIRE_INTERVAL = 60.0 / CYCLIC_RPM  # s between rounds when holding fire

GUN_HALF_EXTENTS = (0.45, 0.04, 0.06)
BULLET_RADIUS = 0.005
BULLET_LENGTH = 0.025

# Anchor offsets in the gun's local frame.
SHOULDER_LOCAL = np.array([-0.35, 0.0, 0.02])
GRIP_LOCAL     = np.array([-0.10, 0.0, -0.05])
HOLD_HEIGHT = 1.5

# Spring-damper "hold" (per anchor). Tuned so a single shot produces a few cm
# of rearward travel and a visible muzzle flip that settles in ~0.3 s.
SHOULDER_K, SHOULDER_C = 2200.0, 90.0
GRIP_K,     GRIP_C     = 900.0,  35.0


def make_gun(start_pos):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=GUN_HALF_EXTENTS)
    vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=GUN_HALF_EXTENTS, rgbaColor=[0.15, 0.15, 0.18, 1]
    )
    body = p.createMultiBody(
        baseMass=GUN_MASS,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=start_pos,
    )
    # Low intrinsic damping — the spring dampers do the work.
    p.changeDynamics(body, -1, linearDamping=0.02, angularDamping=0.05)
    return body


def spawn_bullet(muzzle_pos, forward_dir):
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=BULLET_RADIUS, height=BULLET_LENGTH)
    vis = p.createVisualShape(
        p.GEOM_CYLINDER, radius=BULLET_RADIUS, length=BULLET_LENGTH,
        rgbaColor=[0.95, 0.75, 0.1, 1],
    )
    yaw = math.atan2(forward_dir[1], forward_dir[0])
    pitch = math.asin(max(-1.0, min(1.0, forward_dir[2])))
    orn = p.getQuaternionFromEuler([0, math.pi / 2 - pitch, yaw])

    bullet = p.createMultiBody(
        baseMass=BULLET_MASS,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=muzzle_pos,
        baseOrientation=orn,
    )
    p.changeDynamics(bullet, -1, linearDamping=0.0, angularDamping=0.0)
    p.resetBaseVelocity(bullet, linearVelocity=(forward_dir * MUZZLE_VELOCITY).tolist())
    return bullet


def gun_basis(orn):
    """Return forward/up unit vectors and the rotation matrix."""
    rot = p.getMatrixFromQuaternion(orn)
    R = np.array([[rot[0], rot[1], rot[2]],
                  [rot[3], rot[4], rot[5]],
                  [rot[6], rot[7], rot[8]]])
    return R[:, 0], R[:, 2], R  # local +X forward, local +Z up


def apply_hold_forces(gun_id, rest_shoulder, rest_grip):
    """Spring-damper to each anchor in world space.

    The shoulder anchor resists rearward motion firmly; the grip yields more,
    so the muzzle is free to climb. Damping kills oscillation.
    """
    pos, orn = p.getBasePositionAndOrientation(gun_id)
    lin_v, ang_v = p.getBaseVelocity(gun_id)
    pos = np.array(pos); lin_v = np.array(lin_v); ang_v = np.array(ang_v)
    _, _, R = gun_basis(orn)

    for local, rest, k, c in (
        (SHOULDER_LOCAL, rest_shoulder, SHOULDER_K, SHOULDER_C),
        (GRIP_LOCAL,     rest_grip,     GRIP_K,     GRIP_C),
    ):
        r_world = R @ local                 # COM → anchor in world frame
        anchor_pos = pos + r_world
        v_anchor = lin_v + np.cross(ang_v, r_world)
        F = -k * (anchor_pos - rest) - c * v_anchor
        p.applyExternalForce(gun_id, -1, F.tolist(), anchor_pos.tolist(), p.WORLD_FRAME)


def fire(gun_id):
    """Begin a recoil pulse: spawn the bullet, return (force_vec, world_point, steps)."""
    pos, orn = p.getBasePositionAndOrientation(gun_id)
    forward, up, _ = gun_basis(orn)
    pos = np.array(pos)

    muzzle = pos + forward * (GUN_HALF_EXTENTS[0] + BULLET_LENGTH * 0.6)
    spawn_bullet(muzzle, forward)

    # Total rearward impulse = bullet momentum * ejecta factor.
    impulse_mag = BULLET_MASS * MUZZLE_VELOCITY * EJECTA_FACTOR
    # Convert impulse → constant force across the pulse window.
    force = -forward * (impulse_mag / RECOIL_PULSE)

    # Apply slightly above the bore axis so torque generates muzzle climb,
    # plus a tiny random lateral component for shot-to-shot variation.
    jitter = (random.uniform(-0.002, 0.002), 0.0, random.uniform(0.0, 0.004))
    apply_point = muzzle + up * 0.008 + np.array(jitter)
    return force, apply_point


def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.resetDebugVisualizerCamera(
        cameraDistance=2.2, cameraYaw=55, cameraPitch=-12,
        cameraTargetPosition=[0.2, 0, 1.45],
    )

    dt = 1.0 / 1000.0  # 1 kHz so the 3 ms recoil pulse spans ~3 steps cleanly
    p.setTimeStep(dt)
    p.setPhysicsEngineParameter(numSubSteps=1, fixedTimeStep=dt)

    p.loadURDF("plane.urdf")

    gun_start = np.array([0, 0, HOLD_HEIGHT])
    gun = make_gun(gun_start.tolist())

    # Rest world positions for the two anchors (where the shooter "holds" them).
    rest_shoulder = (gun_start + SHOULDER_LOCAL).copy()
    rest_grip     = (gun_start + GRIP_LOCAL).copy()

    p.addUserDebugText(
        "SPACE: fire    R: reset    ESC: quit",
        [-0.6, 0, 2.0], textColorRGB=[1, 1, 1], textSize=1.2,
    )
    # Visual markers for the rest hold points.
    for pt, color in ((rest_shoulder, [0.2, 0.6, 1]), (rest_grip, [1, 0.4, 0.2])):
        p.addUserDebugLine(pt - [0, 0, 0.02], pt + [0, 0, 0.02], color, 2)

    last_fire = -1e9
    r_was = False
    recoil_force = None
    recoil_point = None
    recoil_steps = 0
    hud_text = -1
    peak_back = 0.0
    peak_pitch = 0.0
    rounds_in_burst = 0
    space_was = False

    sim_steps_per_frame = 4  # render at ~240 fps logical, sim at 1 kHz
    while p.isConnected():
        keys = p.getKeyboardEvents()
        space_now = bool(keys.get(ord(' '), 0) & p.KEY_IS_DOWN)
        r_now     = bool(keys.get(ord('r'), 0) & p.KEY_IS_DOWN)
        now = time.time()

        # Hold space for full-auto at CYCLIC_RPM. A tap fires one round because
        # rising edge always passes the cooldown check (last_fire starts at -inf).
        if space_now and (now - last_fire) >= FIRE_INTERVAL:
            recoil_force, recoil_point = fire(gun)
            recoil_steps = max(1, int(RECOIL_PULSE / dt))
            last_fire = now
            if not space_was:
                rounds_in_burst = 1
                peak_back = 0.0
                peak_pitch = 0.0
            else:
                rounds_in_burst += 1

        if r_now and not r_was:
            p.resetBasePositionAndOrientation(gun, gun_start.tolist(), [0, 0, 0, 1])
            p.resetBaseVelocity(gun, [0, 0, 0], [0, 0, 0])
            recoil_steps = 0
            rounds_in_burst = 0
            peak_back = 0.0
            peak_pitch = 0.0

        for _ in range(sim_steps_per_frame):
            apply_hold_forces(gun, rest_shoulder, rest_grip)
            if recoil_steps > 0:
                p.applyExternalForce(
                    gun, -1, recoil_force.tolist(), recoil_point.tolist(),
                    p.WORLD_FRAME,
                )
                recoil_steps -= 1
            p.stepSimulation()

            # Track peak displacement for the HUD readout.
            pos, orn = p.getBasePositionAndOrientation(gun)
            back = gun_start[0] - pos[0]
            if back > peak_back:
                peak_back = back
            euler = p.getEulerFromQuaternion(orn)
            pitch = euler[1]
            if pitch > peak_pitch:
                peak_pitch = pitch

        if hud_text != -1:
            p.removeUserDebugItem(hud_text)
        hud_text = p.addUserDebugText(
            f"rounds: {rounds_in_burst:>3}   "
            f"peak rearward: {peak_back*100:+.1f} cm   "
            f"peak muzzle rise: {math.degrees(peak_pitch):+.2f} deg",
            [-0.6, 0, 1.85], textColorRGB=[1, 0.7, 0.2], textSize=1.1,
        )

        space_was = space_now
        r_was = r_now
        time.sleep(sim_steps_per_frame * dt)

        if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
            break

    p.disconnect()


if __name__ == "__main__":
    main()
