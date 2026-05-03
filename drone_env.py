"""Gymnasium-compatible RL environment for the drone + machine-gun sim.

Quick usage
-----------

    from drone_env import DroneEnv

    env = DroneEnv(
        gun="m4_carbine",
        n_targets=5,
        wind_mean=(3.0, 0.0, 0.0),
        wind_gust_sigma=1.0,
        recoil_noise=0.04,
        seed=42,
    )

    obs, info = env.reset()
    for _ in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()

Action modes
------------

* ``control_level="setpoint"`` (default) - the on-board cascaded PID flies
  the drone for you. The action is a 5-vector in [-1, 1]:

      [x_des, y_des, z_des, yaw_des, fire]

  with x/y/z/yaw rescaled to physical units and ``fire>0`` meaning trigger
  held this step. Best for high-level RL (waypoint chasing, target picking).

* ``control_level="thrust"`` - the agent commands the four rotor thrusts
  directly. The action is a 5-vector in [-1, 1]:

      [T1, T2, T3, T4, fire]

  with each thrust scaled to [0, max_thrust_per_rotor] and ``fire>0``
  meaning trigger held. Best for low-level RL (learning to hover from
  scratch, attitude control).

Observation
-----------

A flat float32 vector. Components, in order:

      pos              (3)    drone XYZ in world frame
      quat             (4)    drone orientation (w, x, y, z)
      vel              (3)    world-frame linear velocity
      omega_body       (3)    body-frame angular velocity
      ammo_normalized  (1)    rounds remaining / capacity
      wind             (3)    current world-frame wind vector (mean + gust)
      target_rel       (3)    nearest-target position relative to drone
      target_hits      (N)    hit count per target

Reward (default)
----------------

A simple shaped reward intended as a sane starting point:

    r =  10 * new_hits_this_step
       - 0.05 * shots_fired_this_step          (discourage full-auto spam)
       - 0.01 * |nearest target distance|       (encourage proximity)
       - 0.05 * (roll^2 + pitch^2)              (light tilt penalty)
       - 100  if the drone crashes              (terminal)

Subclass and override ``compute_reward(info)`` to do something else (e.g.
sparse rewards, learning to hover, etc.).

Termination
-----------

* z below 0.1 m (crashed)
* roll or pitch beyond 90 deg (flipped)
* horizontal position beyond 30 m (out of bounds)

Truncation: ``max_episode_steps`` (default 500 = 10 s at 50 Hz control rate).
"""
from typing import Optional, Tuple
from dataclasses import asdict

import numpy as np
import mujoco

import gymnasium as gym
from gymnasium import spaces

from config import DRONE, CTRL, SIM, DIST, PROJ
from controller import CascadedController, quat_to_euler
import gun as gun_module
import disturbances as dst
import targets as tgt
import bullets as bul
import casings as cas

# Re-use main.py's model-building helpers (importing main does NOT auto-run
# anything thanks to the `if __name__ == '__main__'` guard there).
from main import build_xml, compute_loadout, get_state, reset_sim, quat_to_rot


class DroneEnv(gym.Env):
    """Quadrotor + machine gun, Gymnasium API."""

    metadata = {"render_modes": ["human", None], "render_fps": 50}

    # =====================================================================
    #                              construction
    # =====================================================================
    def __init__(
        self,
        gun: str = "m4_carbine",
        # ---- scenario ----
        n_targets: int = 5,
        target_radius: float = 5.0,
        target_height: float = 1.5,
        # ---- disturbances ----
        wind_mean: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        wind_gust_sigma: float = 0.0,
        recoil_noise: float = 0.0,
        recoil_angle_noise_deg: float = 0.0,
        # ---- control ----
        control_level: str = "setpoint",
        action_pos_range_m: float = 5.0,
        # ---- optional projectile layers ----
        casings_enabled: bool = False,
        bullets_enabled: bool = False,
        # ---- episode ----
        max_episode_steps: int = 500,
        frame_skip: int = 10,            # physics steps per env.step
        # ---- presentation ----
        render_mode: Optional[str] = None,
        # ---- seeding ----
        seed: Optional[int] = None,
    ):
        super().__init__()
        if control_level not in ("setpoint", "thrust"):
            raise ValueError(f"control_level must be 'setpoint' or 'thrust', got {control_level!r}")

        # --- store config ---
        self.gun_name = gun
        self.n_targets = int(n_targets)
        self.target_radius = float(target_radius)
        self.target_height = float(target_height)
        self.wind_mean = np.asarray(wind_mean, float)
        self.wind_gust_sigma = float(wind_gust_sigma)
        self.recoil_noise = float(recoil_noise)
        self.recoil_angle_noise_deg = float(recoil_angle_noise_deg)
        self.control_level = control_level
        self.action_pos_range_m = float(action_pos_range_m)
        self.casings_enabled = bool(casings_enabled)
        self.bullets_enabled = bool(bullets_enabled)
        self.max_episode_steps = int(max_episode_steps)
        self.frame_skip = int(frame_skip)
        self.render_mode = render_mode
        self._init_seed = seed

        # --- weapon + model ---
        self.weapon = gun_module.make_gun(gun)
        self.total_mass, self.inertia = compute_loadout(self.weapon)
        self.targets_data = (
            tgt.default_ring(n=self.n_targets, radius=self.target_radius,
                             height=self.target_height)
            if self.n_targets > 0 else None
        )
        self._model_xml = build_xml(self.weapon, self.casings_enabled, self.targets_data)
        self.model = mujoco.MjModel.from_xml_string(self._model_xml)
        self.data = mujoco.MjData(self.model)
        self.drone_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drone")

        # --- spaces ---
        # action: 5-vector in [-1, 1] in either control mode (the 5th
        # component is always the fire trigger).
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

        obs_dim = 3 + 4 + 3 + 3 + 1 + 3 + 3 + max(self.n_targets, 0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # --- persistent components, populated in reset() ---
        self.controller = CascadedController(DRONE, CTRL,
                                             total_mass=self.total_mass,
                                             inertia=self.inertia)
        self.target_field: Optional[tgt.TargetField] = None
        self.casing_pool: Optional[cas.CasingPool] = None
        self.bullet_pool: Optional[bul.BulletPool] = None
        self.wind_model: Optional[dst.WindModel] = None

        self._thrust_actual = np.zeros(4)
        self._step_count = 0
        self._viewer = None
        self.rng = np.random.default_rng(seed)

    # =====================================================================
    #                              gym api
    # =====================================================================
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Re-seed everything that's random.
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self._init_seed is not None and not hasattr(self, "_reset_count"):
            # First reset since construction: use the seed passed to __init__.
            self.rng = np.random.default_rng(self._init_seed)
        self._reset_count = getattr(self, "_reset_count", 0) + 1

        # Reset MuJoCo state.
        reset_sim(self.model, self.data)
        self.controller.reset()
        self.weapon.reload()
        hover = self.total_mass * DRONE.gravity / 4.0
        self._thrust_actual[:] = hover
        self.data.xfrc_applied[self.drone_body_id, :] = 0.0
        self._step_count = 0
        self._cum_hits = 0

        # Targets (need re-binding because reset_sim doesn't touch the model).
        if self.targets_data:
            self.target_field = tgt.TargetField(
                self.model, self.data, self.targets_data, self.drone_body_id
            )

        # Casings: re-park the whole pool.
        if self.casings_enabled:
            self.casing_pool = cas.CasingPool(
                self.model, self.data, self.weapon,
                lifetime=PROJ.casing_lifetime_s,
                rng=self.rng,
                prop_collision_warn=False,   # don't spam stdout in RL training
            )

        # Bullets.
        if self.bullets_enabled:
            self.bullet_pool = bul.BulletPool(
                pool_size=PROJ.bullet_pool_size,
                lifetime=PROJ.bullet_lifetime_s,
                gravity=DRONE.gravity,
                drag_const=bul.bullet_drag_const(self.weapon),
                rng=self.rng,
            )

        # Wind.
        self.wind_model = dst.WindModel(
            self.wind_mean, self.wind_gust_sigma, DIST.wind_gust_tau_s, self.rng
        )

        return self._get_obs(), self._info_dict(shots=0, hits=0)

    # ---------------------------------------------------------------------
    def step(self, action):
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        info = {"shots_fired": 0, "hits": 0}
        for _ in range(self.frame_skip):
            self._physics_substep(action, info)
        self._step_count += 1
        self._cum_hits += info["hits"]

        obs = self._get_obs()
        reward = self.compute_reward(info)
        terminated = self._is_terminated()
        truncated = self._step_count >= self.max_episode_steps
        info.update(self._info_dict(shots=info["shots_fired"], hits=info["hits"]))
        return obs, reward, terminated, truncated, info

    # ---------------------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return None
        if self._viewer is None:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self._viewer is not None:
            self._viewer.sync()
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # =====================================================================
    #                         hooks for subclassing
    # =====================================================================
    def compute_reward(self, info) -> float:
        """Default shaped reward. Override in a subclass for a different task."""
        r = 0.0
        r += 10.0 * info["hits"]
        r -= 0.05 * info["shots_fired"]

        state = get_state(self.data)
        if self.target_field is not None:
            nearest = self.target_field.nearest(state["pos"])
            if nearest is not None:
                r -= 0.01 * float(np.linalg.norm(nearest.pos - state["pos"]))

        roll, pitch, _ = quat_to_euler(state["quat"])
        r -= 0.05 * (roll * roll + pitch * pitch)

        if self._is_crashed():
            r -= 100.0
        return float(r)

    # =====================================================================
    #                           internal helpers
    # =====================================================================
    def _physics_substep(self, action, info):
        # ----- decode action -----
        if self.control_level == "thrust":
            thrust_cmd = ((action[:4] + 1.0) * 0.5
                          * DRONE.max_thrust_per_rotor)
            firing = bool(action[4] > 0.0)
        else:  # setpoint
            pos_des = np.array([
                action[0] * self.action_pos_range_m,
                action[1] * self.action_pos_range_m,
                # z stays positive: map [-1, 1] to [0.5, action_pos_range + 0.5]
                (action[2] + 1.0) * 0.5 * self.action_pos_range_m + 0.5,
            ])
            yaw_des = float(action[3] * np.pi)
            firing = bool(action[4] > 0.0)
            state = get_state(self.data)
            sp = {"pos_des": pos_des, "yaw_des": yaw_des}
            thrust_cmd = self.controller.update(state, sp, SIM.timestep)

        # ----- motor lag -----
        alpha = SIM.timestep / (DRONE.motor_tau + SIM.timestep)
        self._thrust_actual += alpha * (thrust_cmd - self._thrust_actual)
        self.data.ctrl[:] = self._thrust_actual

        state = get_state(self.data)

        # ----- wind drag -----
        wind_vec = self.wind_model.step(SIM.timestep)
        F = dst.drag_force(state["vel"], wind_vec, DIST.drone_drag_Cd_A)
        M = np.zeros(3)

        # ----- gun firing -----
        n_shots = self.weapon.step(SIM.timestep, firing)
        if n_shots > 0:
            R = quat_to_rot(state["quat"])
            J_total = 0.0
            jdir_acc = np.zeros(3)
            for _ in range(n_shots):
                factor, jdir = dst.recoil_noise(
                    self.rng, self.weapon.fire_dir_body,
                    self.recoil_noise, self.recoil_angle_noise_deg,
                )
                J_one = factor * self.weapon.impulse_per_shot_Ns
                J_total += J_one
                jdir_acc += J_one * jdir
            fire_dir = jdir_acc / J_total
            F_body = -fire_dir * (J_total / SIM.timestep)
            r_body = np.asarray(self.weapon.mount_offset_m, float)
            F = F + R @ F_body
            M = M + R @ np.cross(r_body, F_body)

            muzzle_world = state["pos"] + R @ (
                np.asarray(self.weapon.mount_offset_m, float)
                + np.asarray(self.weapon.fire_dir_body, float) * 0.30
            )
            fire_dir_world = R @ fire_dir

            if self.casing_pool is not None:
                self.casing_pool.eject(
                    n_shots, state["pos"], R, state["vel"],
                    speed_sigma=DIST.casing_eject_speed_sigma,
                    angle_sigma_deg=DIST.casing_eject_angle_sigma_deg,
                )
            if self.bullet_pool is not None:
                for _ in range(n_shots):
                    v_world = state["vel"] + R @ (self.weapon.muzzle_vel_mps * fire_dir)
                    self.bullet_pool.spawn(muzzle_world, v_world)

            if self.target_field is not None:
                for _ in range(n_shots):
                    hits = self.target_field.cast_shot(
                        muzzle_world, fire_dir_world,
                        n_pellets=self.weapon.pellets_per_shot,
                        spread_deg=self.weapon.pellet_spread_deg,
                        rng=self.rng,
                    )
                    info["hits"] += len(hits)

            info["shots_fired"] += n_shots

        self.data.xfrc_applied[self.drone_body_id, 0:3] = F
        self.data.xfrc_applied[self.drone_body_id, 3:6] = M
        mujoco.mj_step(self.model, self.data)

        if self.casing_pool is not None:
            self.casing_pool.step(SIM.timestep)
            self.casing_pool.check_prop_hits()
        if self.bullet_pool is not None:
            self.bullet_pool.step(SIM.timestep)

    # ---------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        state = get_state(self.data)
        wind_now = (self.wind_model.mean + self.wind_model.gust
                    if self.wind_model is not None else np.zeros(3))

        if self.target_field is not None and self.target_field.targets:
            nearest = self.target_field.nearest(state["pos"])
            target_rel = (nearest.pos - state["pos"]) if nearest is not None else np.zeros(3)
            hits = np.array([t.hits for t in self.target_field.targets], dtype=float)
        else:
            target_rel = np.zeros(3)
            hits = np.zeros(self.n_targets, dtype=float) if self.n_targets > 0 else np.zeros(0)

        ammo_norm = self.weapon.ammo / max(1, self.weapon.capacity_rounds)

        return np.concatenate([
            state["pos"], state["quat"], state["vel"], state["omega_body"],
            np.array([ammo_norm], dtype=float),
            wind_now,
            target_rel,
            hits,
        ]).astype(np.float32)

    def _info_dict(self, shots, hits) -> dict:
        return {
            "shots_fired": shots,
            "hits": hits,
            "cumulative_hits": self._cum_hits,
            "ammo": int(self.weapon.ammo),
            "step_count": self._step_count,
        }

    # ---------------------------------------------------------------------
    def _is_crashed(self) -> bool:
        return self.data.qpos[2] < 0.1

    def _is_terminated(self) -> bool:
        if self._is_crashed():
            return True
        roll, pitch, _ = quat_to_euler(self.data.qpos[3:7])
        if abs(roll) > np.pi / 2 or abs(pitch) > np.pi / 2:
            return True
        x, y, z = self.data.qpos[0:3]
        if abs(x) > 30 or abs(y) > 30 or z > 30:
            return True
        return False
