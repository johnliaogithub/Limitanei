import drone_env as drone_module
import numpy as np

from controller import quat_to_euler
from main import get_state, quat_to_rot

class SimpleRewardEnv(drone_module.DroneEnv): 
    """
    Single target environment with M249 drone

    Initialized 100 meters from target

    Environment with dense reward: 
     - hit target
     - stable: penalize drone sideways rotation above 10 degrees
     - distance between line and target center = norm of perpendicular of the target vector onto the gun direction
    """

    EPS = 1e-7

    def __init__(self, **kwargs):
        kwargs.setdefault("n_targets", 1)
        kwargs.setdefault("target_radius", 100.0)
        kwargs.setdefault("target_height", 1.5)
        kwargs.setdefault("action_pos_range_m", 110.0)
        super().__init__(**kwargs)

    def compute_reward(self, info) -> float: 
        r = 0.0
        r += 10.0 * info["hits"]
        r -= 0.05 * info["shots_fired"]

        if info["shots_fired"] > 0: 
            # find distance between gun direction and target
            state = get_state(self.data)
            R = quat_to_rot(state["quat"])
            gun_dir_world = R @ np.asarray(self.weapon.fire_dir_body, float)
            muzzle_world = state["pos"] + R @ (
                np.asarray(self.weapon.mount_offset_m, float)
                + np.asarray(self.weapon.fire_dir_body, float) * 0.30   # barrel length
            )
            v = target_loc - muzzle_world
            perp = v - np.dot(v, gun_dir_world) * gun_dir_world          # gun_dir_world is unit
            miss_distance = float(np.linalg.norm(perp))

            r += np.clip(0.05 / (miss_distance + self.EPS), 0, 1)  # less than or equal to 10 centimeters away from the target center is reward of 1

        # penalize instability on the roll axis (pitch and yaw are relatively unimportant, since roll seems to destabilize the drone more in PID)
        roll, pitch, _ = quat_to_euler(state["quat"])
        pitch_excess = max(0.0, abs(pitch) - np.pi / 3)   # zero up to 60°, then linear. 60% is fine because the drone may need to aim downwards
        r -= 0.05 * (roll**2 + 0.5 * pitch_excess**2)

        if self._is_crashed():
            r -= 100.0

        return float(r)
