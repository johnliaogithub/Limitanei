"""Quadrotor MuJoCo simulation - entry point.

Examples:

    # Default autonomous demo (M4 carbine):
    python main.py --mode auto

    # Pilot manually with a different weapon:
    python main.py --mode keyboard --gun pkm

    # Repeatable run with realistic disturbances and live casings/bullets:
    python main.py --gun aa12_shotgun --seed 42 --recoil-noise 0.04 \
                   --wind 5 0 0 --gust 1.5 --projectiles

    # Just list every weapon's recoil profile:
    python main.py --list-guns
"""
import argparse
import time
import numpy as np

import mujoco
import mujoco.viewer

from drone_sim.config import DRONE, CTRL, SIM, GUN, DIST, PROJ
from drone_sim.control.controller import CascadedController, quat_to_euler
from drone_sim.control.modes import KeyboardSetpoint, AutonomousSetpoint
from drone_sim.physics import gun as gun_module
from drone_sim.physics import disturbances as dst
from drone_sim.physics import bullets as bul
from drone_sim.physics import casings as cas
from drone_sim.physics import targets as tgt
from drone_sim.io.logger import TrajectoryRecorder
from dataclasses import asdict


# ----------------------------------------------------------------------------
#  Build the MuJoCo XML model from the config + selected gun loadout.
# ----------------------------------------------------------------------------
def compute_loadout(weapon: gun_module.Gun):
    """Return (total_mass, (Ixx, Iyy, Izz)) for the airframe + weapon system.

    The weapon is treated as a point mass at its mount offset, so its
    rotational inertia about the drone's CoM is just the parallel-axis term
    m * (perpendicular distance)^2 along each axis.
    """
    m_g = weapon.loaded_mass_kg
    mx, my, mz = weapon.mount_offset_m
    total_mass = DRONE.mass + m_g
    Ixx = DRONE.Ixx + m_g * (my * my + mz * mz)
    Iyy = DRONE.Iyy + m_g * (mx * mx + mz * mz)
    Izz = DRONE.Izz + m_g * (mx * mx + my * my)
    return total_mass, (Ixx, Iyy, Izz)


def build_xml(weapon: gun_module.Gun, casings_on: bool,
              targets_data=None) -> str:
    L = DRONE.arm_length / np.sqrt(2)         # X-config offset along x and y
    cq = DRONE.k_q_over_k_t                   # gear[5] = yaw drag torque per unit thrust
    Tmax = DRONE.max_thrust_per_rotor
    px, py, pz = SIM.init_pos

    total_mass, (Ixx, Iyy, Izz) = compute_loadout(weapon)
    mx, my, mz = weapon.mount_offset_m
    barrel_len = 0.30                         # m, cosmetic

    # contype/conaffinity bitmasks (see projectiles.py for the full table).
    # bit 1 = floor, bit 2 = drone airframe, bit 4 = prop discs, bit 8 = casings.
    if casings_on:
        # Drone airframe collides with floor only (bit 1).
        airframe_ct = 'contype="2" conaffinity="1"'
        prop_disc_xml = cas.prop_disc_geom_xml(L)
        casing_pool_xml = cas.casing_pool_xml(PROJ.casing_pool_size, weapon)
    else:
        airframe_ct = ''
        prop_disc_xml = ''
        casing_pool_xml = ''
    target_xml_str = tgt.target_xml(targets_data) if targets_data else ''

    return f"""
<mujoco model="quadrotor_armed">
  <option timestep="{SIM.timestep}" gravity="0 0 -{DRONE.gravity}" integrator="RK4"
          density="1.225" viscosity="1.8e-5"/>
  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.3 0.3 0.3"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global offwidth="1280" offheight="800" elevation="-25" azimuth="135"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker"
             rgb1="0.30 0.40 0.30" rgb2="0.40 0.50 0.40" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="10 10"/>
  </asset>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="40 40 0.1" material="grid"/>

    <body name="drone" pos="{px} {py} {pz}">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="{total_mass}"
                diaginertia="{Ixx} {Iyy} {Izz}"/>

      <!-- airframe (cosmetic + collision shape) -->
      <geom name="core" type="box" size="0.07 0.07 0.025" rgba="0.15 0.15 0.15 1" {airframe_ct}/>
      <geom type="capsule" fromto="0 0 0  {L}  {-L} 0" size="0.012" rgba="0.3 0.3 0.3 1" {airframe_ct}/>
      <geom type="capsule" fromto="0 0 0  {L}   {L} 0" size="0.012" rgba="0.3 0.3 0.3 1" {airframe_ct}/>
      <geom type="capsule" fromto="0 0 0 {-L}   {L} 0" size="0.012" rgba="0.3 0.3 0.3 1" {airframe_ct}/>
      <geom type="capsule" fromto="0 0 0 {-L}  {-L} 0" size="0.012" rgba="0.3 0.3 0.3 1" {airframe_ct}/>

      <!-- rotor markers (sites = thrust application points, no collision) -->
      <site name="rotor1" pos=" {L} {-L} 0.025" size="0.08 0.005" type="cylinder" rgba="1.0 0.3 0.3 0.7"/>
      <site name="rotor2" pos=" {L}  {L} 0.025" size="0.08 0.005" type="cylinder" rgba="0.3 1.0 0.3 0.7"/>
      <site name="rotor3" pos="{-L}  {L} 0.025" size="0.08 0.005" type="cylinder" rgba="0.3 0.3 1.0 0.7"/>
      <site name="rotor4" pos="{-L} {-L} 0.025" size="0.08 0.005" type="cylinder" rgba="1.0 1.0 0.3 0.7"/>
      {prop_disc_xml}

      <!-- gun barrel + muzzle (cosmetic only - recoil is applied via xfrc_applied). -->
      <geom name="gun" type="capsule" size="0.018"
            fromto="{mx} {my} {mz}  {mx + barrel_len} {my} {mz}"
            rgba="0.1 0.1 0.1 1" {airframe_ct}/>
      <site name="muzzle" pos="{mx + barrel_len} {my} {mz}" size="0.03" rgba="1 0.6 0 0.9"/>
    </body>
    {casing_pool_xml}
    {target_xml_str}
  </worldbody>

  <actuator>
    <motor name="m1" site="rotor1" gear="0 0 1 0 0  {cq}" ctrlrange="0 {Tmax}"/>
    <motor name="m2" site="rotor2" gear="0 0 1 0 0 {-cq}" ctrlrange="0 {Tmax}"/>
    <motor name="m3" site="rotor3" gear="0 0 1 0 0  {cq}" ctrlrange="0 {Tmax}"/>
    <motor name="m4" site="rotor4" gear="0 0 1 0 0 {-cq}" ctrlrange="0 {Tmax}"/>
  </actuator>
</mujoco>
"""


def get_state(data) -> dict:
    """Extract the drone state from the MuJoCo data struct.
    For a free joint:  qpos = [x,y,z, qw,qx,qy,qz]   (linear pos + orientation)
                       qvel = [vx,vy,vz, wx,wy,wz]   (lin vel WORLD, ang vel BODY)
    """
    return {
        'pos':         data.qpos[0:3].copy(),
        'quat':        data.qpos[3:7].copy(),
        'vel':         data.qvel[0:3].copy(),
        'omega_body':  data.qvel[3:6].copy(),
    }


def reset_sim(model, data):
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = SIM.init_pos
    half = SIM.init_yaw * 0.5
    data.qpos[3:7] = [np.cos(half), 0.0, 0.0, np.sin(half)]
    mujoco.mj_forward(model, data)


def quat_to_rot(q):
    """MuJoCo quat [w, x, y, z] -> 3x3 rotation matrix R (body -> world)."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


# ----------------------------------------------------------------------------
#  Main loop
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Quadrotor + machine-gun MuJoCo sim")
    parser.add_argument('--mode', choices=['keyboard', 'auto'], default='auto')
    parser.add_argument('--gun', default=GUN.selected,
                        help=f"weapon model (one of: {', '.join(gun_module.GUNS)})")
    parser.add_argument('--list-guns', action='store_true')
    parser.add_argument('--debug-keys', action='store_true',
                        help="print every key event (vk/char/decoded token)")
    parser.add_argument('--seed', type=int, default=DIST.seed,
                        help="RNG seed for reproducibility (0 = system entropy)")
    parser.add_argument('--bullets', action='store_true',
                        help="enable Python-side bullet ballistics + tracer rendering")
    parser.add_argument('--casings', action='store_true',
                        help="enable MuJoCo casing physics + propeller-impact detection")
    parser.add_argument('--wind', type=float, nargs=3, metavar=('WX', 'WY', 'WZ'),
                        default=list(DIST.wind_mean_mps),
                        help="mean wind in m/s, world frame")
    parser.add_argument('--gust', type=float, default=DIST.wind_gust_sigma_mps,
                        help="wind gust std-dev in m/s (Ornstein-Uhlenbeck)")
    parser.add_argument('--recoil-noise', type=float, default=DIST.recoil_impulse_sigma,
                        help="per-shot recoil impulse std dev (e.g. 0.04 = 4%%)")
    parser.add_argument('--recoil-angle-noise', type=float, default=DIST.recoil_angular_sigma_deg,
                        help="per-shot firing-direction wobble std dev (deg)")
    parser.add_argument('--targets', type=int, default=0,
                        help="number of targets in a ring around the origin (0 = none)")
    parser.add_argument('--target-radius', type=float, default=5.0,
                        help="ring radius for --targets (m)")
    parser.add_argument('--target-height', type=float, default=1.5,
                        help="ring height for --targets (m)")
    parser.add_argument('--aim-yaw', action='store_true',
                        help="autonomous mode yaws to face the nearest target")
    parser.add_argument('--record', metavar='PATH', default=None,
                        help="record trajectory + events to this .npz for later replay/analysis")
    parser.add_argument('--record-decimation', type=int, default=5,
                        help="record every Nth physics step (5 → 100 Hz log on a 500 Hz sim)")
    args = parser.parse_args()

    if args.list_guns:
        for k, v in gun_module.GUNS.items():
            g = gun_module.make_gun(k)
            print(f"  {k:14s}  {v['name']}")
            print(f"  {'':14s}    impulse/shot = {g.impulse_per_shot_Ns:5.2f} N*s, "
                  f"avg recoil @ {g.cyclic_rate_rpm:.0f} RPM = {g.avg_recoil_force_N:6.1f} N, "
                  f"loaded mass = {g.loaded_mass_kg:.2f} kg")
        return

    # ---- RNG (the SOLE source of randomness for the whole sim) ----
    rng_seed = args.seed if args.seed != 0 else None
    rng = np.random.default_rng(rng_seed)
    print(f"[main] RNG seed = {args.seed} (0 means system entropy)")

    # ---- Build model + data ----
    weapon = gun_module.make_gun(args.gun)
    total_mass, inertia = compute_loadout(weapon)
    PROJ.enabled = args.bullets or args.casings
    targets_data = None
    if args.targets > 0:
        targets_data = tgt.default_ring(n=args.targets,
                                        radius=args.target_radius,
                                        height=args.target_height)
    model_xml_str = build_xml(weapon, args.casings, targets_data)
    model = mujoco.MjModel.from_xml_string(model_xml_str)
    data = mujoco.MjData(model)
    drone_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'drone')
    reset_sim(model, data)

    target_field = None
    if targets_data:
        target_field = tgt.TargetField(model, data, targets_data, drone_body_id)
        print(f"[main] Targets: {len(target_field.targets)} in a ring of "
              f"r={args.target_radius:.1f} m at z={args.target_height:.1f} m")

    controller = CascadedController(DRONE, CTRL,
                                    total_mass=total_mass, inertia=inertia)
    if args.mode == 'keyboard':
        setpoint_gen = KeyboardSetpoint(CTRL, SIM, debug_keys=args.debug_keys)
    else:
        setpoint_gen = AutonomousSetpoint(
            SIM, auto_fire=GUN.auto_fire_in_auto_mode,
            target_field=target_field, aim_yaw_at_target=args.aim_yaw)

    # ---- Disturbances ----
    wind = dst.WindModel(mean_mps=args.wind, sigma_mps=args.gust,
                         tau_s=DIST.wind_gust_tau_s, rng=rng)

    # ---- Optional projectile layers (independently toggleable) ----
    casings = bullets = None
    if args.casings:
        casings = cas.CasingPool(model, data, weapon,
                                 lifetime=PROJ.casing_lifetime_s,
                                 rng=rng,
                                 prop_collision_warn=PROJ.prop_collision_warn)
        print(f"[main] Casing mode: {casings.n} pool slots")
    if args.bullets:
        bullets = bul.BulletPool(pool_size=PROJ.bullet_pool_size,
                                 lifetime=PROJ.bullet_lifetime_s,
                                 gravity=DRONE.gravity,
                                 drag_const=bul.bullet_drag_const(weapon),
                                 rng=rng)
        print(f"[main] Bullet mode: {PROJ.bullet_pool_size} tracer slots")

    hover_thrust = total_mass * DRONE.gravity / 4.0
    thrust_cmd = np.full(4, hover_thrust)
    thrust_actual = np.full(4, hover_thrust)

    ctrl_period = 1.0 / SIM.control_hz
    last_ctrl_t = -1.0

    # ---- Trajectory recorder (optional) ----
    recorder = None
    if args.record is not None:
        recorder = TrajectoryRecorder(args.record, decimation=args.record_decimation)
        print(f"[main] Recording to {args.record} every {args.record_decimation} steps "
              f"(~{SIM.timestep * args.record_decimation * 1000:.1f} ms / sample)")

    print("[main] " + "=" * 60)
    print(f"[main] Mode = {args.mode}, Weapon = {weapon.name}")
    print(f"[main] Drone mass {DRONE.mass:.2f} kg + weapon+ammo {weapon.loaded_mass_kg:.2f} kg "
          f"= {total_mass:.2f} kg total")
    print(f"[main] Hover thrust per rotor = {hover_thrust:.2f} N "
          f"(motor max {DRONE.max_thrust_per_rotor:.1f} N)")
    print(f"[main] Recoil: J = {weapon.impulse_per_shot_Ns:.2f} N*s/shot, "
          f"avg force at {weapon.cyclic_rate_rpm:.0f} RPM = {weapon.avg_recoil_force_N:.1f} N "
          f"({weapon.avg_recoil_force_N / (total_mass * DRONE.gravity) * 100:.1f}% of weight)")
    print(f"[main] Wind: mean {tuple(args.wind)} m/s, gust sigma {args.gust} m/s")
    print(f"[main] Recoil noise: impulse sigma {args.recoil_noise:.2%}, "
          f"angle sigma {args.recoil_angle_noise:.2f} deg")
    if args.mode == 'keyboard':
        print("[main] NumLock ON. Numpad layout:")
        print("[main]   7 yaw-L | 8 fwd  | 9 yaw-R")
        print("[main]   4 left  | 5 RST  | 6 right")
        print("[main]             2 back")
        print("[main]   + climb           - descend")
        print("[main]   0 = FIRE")
    print("[main] " + "=" * 60)

    def do_reset():
        nonlocal last_ctrl_t
        reset_sim(model, data)
        controller.reset()
        setpoint_gen.reset()
        weapon.reload()
        thrust_cmd[:] = hover_thrust
        thrust_actual[:] = hover_thrust
        data.xfrc_applied[drone_body_id, :] = 0.0
        last_ctrl_t = -1.0
        if casings is not None:
            for i in range(casings.n):
                casings._park(i)
        if bullets is not None:
            bullets.life[:] = 0.0
        if target_field is not None:
            target_field.reset()
        if recorder is not None:
            recorder.event(data.time, "reset")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t_step_start = time.time()
            t = data.time

            # Reset triggers: explicit Numpad-5 OR sim-time jumped backward
            # (which happens when the user hits Backspace in the viewer).
            if args.mode == 'keyboard' and getattr(setpoint_gen, 'reset_flag', False):
                setpoint_gen.reset_flag = False
                do_reset()
                t = data.time
            elif last_ctrl_t > t:
                do_reset()
                t = data.time

            # ---- Run controller at SIM.control_hz ----
            if last_ctrl_t < 0.0 or (t - last_ctrl_t) >= ctrl_period:
                dt_ctrl = ctrl_period if last_ctrl_t >= 0.0 else SIM.timestep
                state = get_state(data)
                _, _, yaw_now = quat_to_euler(state['quat'])
                sp = setpoint_gen.step(dt_ctrl, yaw_now)
                thrust_cmd = controller.update(state, sp, dt_ctrl)
                last_ctrl_t = t
            else:
                state = get_state(data)

            # ---- Motor first-order lag ----
            alpha = SIM.timestep / (DRONE.motor_tau + SIM.timestep)
            thrust_actual += alpha * (thrust_cmd - thrust_actual)
            data.ctrl[:] = thrust_actual

            # ---- Disturbance: wind drag ----
            wind_vec = wind.step(SIM.timestep)
            F_drag = dst.drag_force(state['vel'], wind_vec, DIST.drone_drag_Cd_A)
            F_total_world = F_drag.copy()
            M_total_world = np.zeros(3)

            # ---- Gun: count rounds fired this step, compute recoil + spawn projectiles ----
            firing = setpoint_gen.is_firing()
            n_shots = weapon.step(SIM.timestep, firing)
            if n_shots > 0:
                R = quat_to_rot(state['quat'])
                # Per-shot impulse with optional Gaussian magnitude noise; the
                # average over many shots converges to the nominal value, so
                # the controller's steady-state behaviour is unchanged.
                J_total = 0.0
                jitter_dir_avg = np.zeros(3)
                for _ in range(n_shots):
                    factor, jdir = dst.recoil_noise(rng, weapon.fire_dir_body,
                                                    args.recoil_noise,
                                                    args.recoil_angle_noise)
                    J_one = factor * weapon.impulse_per_shot_Ns
                    J_total += J_one
                    jitter_dir_avg += J_one * jdir
                fire_dir = jitter_dir_avg / J_total
                F_body = -fire_dir * (J_total / SIM.timestep)
                r_body = np.asarray(weapon.mount_offset_m, dtype=float)
                M_body = np.cross(r_body, F_body)
                F_total_world += R @ F_body
                M_total_world += R @ M_body

                # Spawn casings + bullets
                muzzle_world = state['pos'] + R @ (
                    np.asarray(weapon.mount_offset_m, float)
                    + np.asarray(weapon.fire_dir_body, float) * 0.30)
                if casings is not None:
                    casings.eject(n_shots, state['pos'], R, state['vel'],
                                  speed_sigma=DIST.casing_eject_speed_sigma,
                                  angle_sigma_deg=DIST.casing_eject_angle_sigma_deg)
                if bullets is not None:
                    for _ in range(n_shots):
                        v_world = state['vel'] + R @ (weapon.muzzle_vel_mps * fire_dir)
                        bullets.spawn(muzzle_world, v_world)

                # Ray-cast hit detection on targets. Each shot fires
                # `pellets_per_shot` rays (1 for rifles/SMGs, 9 for the AA-12)
                # and each ray gets independent angular jitter.
                if target_field is not None:
                    fire_dir_world = R @ fire_dir
                    for _ in range(n_shots):
                        hits = target_field.cast_shot(
                            muzzle_world, fire_dir_world,
                            n_pellets=weapon.pellets_per_shot,
                            spread_deg=weapon.pellet_spread_deg,
                            rng=rng)
                        for t, hp in hits:
                            print(f"[hit] target_{t.id} at "
                                  f"({hp[0]:+.2f}, {hp[1]:+.2f}, {hp[2]:+.2f}) "
                                  f"[{t.hits}/{t.max_hits}]")
                            if recorder is not None:
                                recorder.event(data.time, "target_hit",
                                               target_id=t.id, hit_pos=hp,
                                               hits=t.hits)

                # Log the shot itself
                if recorder is not None:
                    recorder.event(data.time, "shot",
                                   n_shots=n_shots,
                                   muzzle_pos=muzzle_world,
                                   fire_dir_world=(R @ fire_dir).tolist(),
                                   impulse_total=J_total)

            data.xfrc_applied[drone_body_id, 0:3] = F_total_world
            data.xfrc_applied[drone_body_id, 3:6] = M_total_world

            mujoco.mj_step(model, data)

            # ---- Projectile bookkeeping (after physics step) ----
            if casings is not None:
                casings.step(SIM.timestep)
                n_prop_hits = casings.check_prop_hits()
                if n_prop_hits > 0 and recorder is not None:
                    recorder.event(data.time, "prop_hit", n=n_prop_hits)
            if bullets is not None:
                bullets.step(SIM.timestep)
                viewer.user_scn.ngeom = 0
                bullets.render(viewer.user_scn)

            # ---- Trajectory record ----
            if recorder is not None:
                recorder.record(
                    data.time, data,
                    pos_des=sp['pos_des'],
                    yaw_des=sp['yaw_des'],
                    wind=wind_vec,
                    ammo=weapon.ammo,
                    xfrc_drone=data.xfrc_applied[drone_body_id, :].copy(),
                )

            viewer.sync()

            if SIM.realtime:
                slack = SIM.timestep - (time.time() - t_step_start)
                if slack > 0:
                    time.sleep(slack)

    # ---- Persist the run when the viewer closes ----
    if recorder is not None:
        metadata = {
            "model_xml": model_xml_str,
            "gun_name": args.gun,
            "cli_args": {k: v for k, v in vars(args).items()
                         if isinstance(v, (int, float, str, bool, list, tuple, type(None)))},
            "config": {
                "DRONE": asdict(DRONE),
                "CTRL": asdict(CTRL),
                "SIM": asdict(SIM),
                "DIST": asdict(DIST),
                "PROJ": asdict(PROJ),
            },
        }
        recorder.save(metadata)


if __name__ == '__main__':
    main()
