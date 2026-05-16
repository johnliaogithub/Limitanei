import numpy as np

from drone_sim.rl import env as drone_module
from drone_sim.control.controller import quat_to_euler
from drone_sim.sim import get_state, quat_to_rot
from drone_sim.config import SIM


# Default reward coefficients — override any subset via reward_cfg={...}
_ZERO_DEFAULTS = dict(
    uprightness  = 1.0, # 1.0,   # cos(roll)*cos(pitch) bonus for being level
    omega        = 0.1,   # penalty on squared angular velocity magnitude
    altitude     = 0.1,   # penalty on squared altitude deviation from spawn
    horiz_vel    = 0.5,   # penalty coefficient when |vel_xy| exceeds threshold
    horiz_thresh = 1.0,   # m/s — horizontal drift up to this is free
    survival     = 0.1,   # reward for surviving each step
    crash        = 100.0, # one-time penalty on crash
)

_SINGLE_DEFAULTS = dict(
    hit          = 30.0,  # reward per bullet hit
    aim          = 0.3,   # always-on cosine aiming reward (max per step)
    miss_dist    = 0.05,  # numerator of 1/miss_distance shot-shaping bonus
    ammo         = 0.005, # penalty per shot fired
    stability    = 0.01,  # coefficient on roll² + 0.5*pitch_excess²
    crash        = 75.0,  # one-time penalty on crash
    min_dist_m   = 0.0,   # minimum approach distance (m); 0 = disabled
    min_dist_pen = 2.0,   # quadratic penalty coefficient when inside min_dist_m
    eff_range_m  = 0.0,   # effective-range zone radius (m); 0 = disabled
    eff_range    = 0.2,   # flat bonus per step when inside eff_range_m
)


class ZeroTargetEnv(drone_module.DroneEnv):
    """Stabilisation-only environment (no target).

    Dense reward: stay upright, maintain altitude, limit horizontal drift.
    Horizontal velocity up to horiz_thresh m/s is free; beyond that a
    quadratic penalty kicks in, teaching the policy to decelerate without
    requiring it to hold a fixed x,y position.

    reward_cfg keys (all optional, merged with defaults):
        uprightness, omega, altitude, horiz_vel, horiz_thresh, crash
    """

    EPS = 1e-7

    def __init__(self, reward_cfg: dict | None = None, **kwargs):
        kwargs.setdefault("n_targets", 0)
        kwargs.setdefault("gun", "hk416")      # must match BC training weapon
        kwargs.setdefault("obs_pos_xy", False)
        super().__init__(**kwargs)
        cfg = dict(_ZERO_DEFAULTS)
        if reward_cfg:
            cfg.update(reward_cfg)
        self._rc = cfg

    def _get_obs(self):
        return super()._get_obs()

    def compute_reward(self, _info) -> float:
        r = 0.0
        rc = self._rc

        state = get_state(self.data)
        roll, pitch, _ = quat_to_euler(state["quat"])
        omega = state["omega_body"]

        # perhaps uprightness not as important
        r += rc["uprightness"] * np.cos(roll) * np.cos(pitch)
        r -= rc["omega"]       * float(np.dot(omega, omega))
        r -= rc["altitude"]    * (state["pos"][2] - SIM.init_pos[2]) ** 2

        # Penalise horizontal velocity only when it exceeds the free threshold.
        vel_h  = float(np.linalg.norm(state["vel"][:2]))
        excess = max(0.0, vel_h - rc["horiz_thresh"])
        r -= rc["horiz_vel"] * excess ** 2

        if self._is_crashed():
            r -= rc["crash"]
        else: 
            # reward survival
            r += rc['survival']

        return float(r)


class SingleTargetEnv(drone_module.DroneEnv):
    """Single-target shooting environment.

    Dense reward: aim at target, fire, hit, stay stable.

    reward_cfg keys (all optional, merged with defaults):
        hit          — reward per bullet hit
        aim          — always-on cosine aiming bonus (max per step)
        miss_dist    — shot-shaping numerator: clip(miss_dist / d_perp, 0, 1)
        ammo         — penalty per shot fired
        stability    — coefficient on roll² + 0.5*pitch_excess²
        crash        — one-time crash penalty
        min_dist_m   — minimum safe approach distance in metres (0 = disabled)
        min_dist_pen — quadratic penalty coefficient when closer than min_dist_m
        eff_range_m  — effective-range zone radius in metres (0 = disabled)
        eff_range    — flat reward per step when inside eff_range_m
    """

    EPS = 1e-7

    def __init__(self, reward_cfg: dict | None = None, **kwargs):
        kwargs.setdefault("n_targets", 1)
        kwargs.setdefault("gun", "hk416")
        kwargs.setdefault("target_radius", 100.0)
        kwargs.setdefault("target_height", 1.5)
        kwargs.setdefault("action_pos_range_m", 110.0)
        kwargs.setdefault("obs_pos_xy", False)
        # Face the target at spawn so the aiming reward kicks in immediately.
        # Switch to "random" once the policy reliably hits.
        kwargs.setdefault("spawn_heading", "toward_target")
        super().__init__(**kwargs)
        cfg = dict(_SINGLE_DEFAULTS)
        if reward_cfg:
            cfg.update(reward_cfg)
        self._rc = cfg

    def compute_reward(self, info) -> float:
        r = 0.0
        rc = self._rc

        r += rc["hit"] * info["hits"]

        state = get_state(self.data)
        R = quat_to_rot(state["quat"])
        gun_dir_world = R @ np.asarray(self.weapon.fire_dir_body, float)
        target_loc = self.target_field.targets[0].pos
        v_to_target = target_loc - state["pos"]
        dist = float(np.linalg.norm(v_to_target))

        # Always-on angular aiming reward: 1.0 when gun points exactly at
        # target, 0 when perpendicular. Provides dense gradient even before
        # the policy learns to fire, preventing "never shoot" local optima.
        if dist > 0.5:
            cos_angle = float(np.dot(gun_dir_world, v_to_target / dist))
            r += rc["aim"] * np.clip(cos_angle, 0.0, 1.0)

        if info["shots_fired"] > 0:
            muzzle_world = state["pos"] + R @ (
                np.asarray(self.weapon.mount_offset_m, float)
                + np.asarray(self.weapon.fire_dir_body, float) * 0.30
            )
            v = target_loc - muzzle_world
            perp = v - np.dot(v, gun_dir_world) * gun_dir_world
            miss_distance = float(np.linalg.norm(perp))
            r += np.clip(rc["miss_dist"] / (miss_distance + self.EPS), 0.0, 1.0)
            r -= rc["ammo"] * info["shots_fired"]

        roll, pitch, _ = quat_to_euler(state["quat"])
        pitch_excess = max(0.0, abs(pitch) - np.pi / 3)
        r -= rc["stability"] * (roll ** 2 + 0.5 * pitch_excess ** 2)

        if rc["eff_range_m"] > 0 and dist < rc["eff_range_m"]:
            r += rc["eff_range"]

        if rc["min_dist_m"] > 0 and dist < rc["min_dist_m"]:
            r -= rc["min_dist_pen"] * (rc["min_dist_m"] - dist) ** 2

        if self._is_crashed():
            r -= rc["crash"]

        return float(r)
