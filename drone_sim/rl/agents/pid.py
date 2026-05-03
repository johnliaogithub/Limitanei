"""Heuristic PID-style agent for SimpleRewardEnv.

The env's setpoint control mode already runs a cascaded PID flight controller
internally - this 'agent' just decides each step's high-level goal:

    1. Hold a fixed hover position.
    2. Yaw to face the nearest target (proportional in yaw error).
    3. Fire when |yaw error| is below a threshold.

It's a deterministic baseline. Useful for sanity-checking the env's reward
shape and as a comparison point for any learned agent you train later.

Run:
    python pid_agent.py                  # headless, plot at end
    python pid_agent.py --render         # also opens the MuJoCo viewer
    python pid_agent.py --gun aa12_shotgun --max-steps 1000
"""
import argparse
import time
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from drone_sim.control.controller import quat_to_euler, wrap_pi
from drone_sim.rl.custom_envs import SimpleRewardEnv


# Observation layout (drone_env.DroneEnv._get_obs):
#   pos          [0:3]    quat         [3:7]
#   vel          [7:10]   omega_body   [10:13]
#   ammo_norm    [13]     wind         [14:17]
#   target_rel   [17:20]  target_hits  [20:]
def _decode_obs(obs):
    return {
        "pos": obs[0:3],
        "quat": obs[3:7],
        "vel": obs[7:10],
        "omega_body": obs[10:13],
        "ammo": obs[13],
        "target_rel": obs[17:20],
        "hits": obs[20:],
    }


class PIDAgent:
    """Hover at a fixed point, yaw-track the target, fire when aligned."""

    def __init__(self, hover_pos=(0.0, 0.0, 1.55),
                 fire_yaw_threshold_deg: float = 3.0,
                 fire_tilt_threshold_deg: float = 8.0,
                 fire_omega_threshold_rps: float = 1.0,
                 fire_pos_threshold_m: float = 0.25,
                 action_pos_range_m: float = 5.0):
        # NB: hover_pos default z is 1.55 so the gun's muzzle (which sits 5 cm
        # below the drone's CoM thanks to mount_offset_m) ends up at z=1.50,
        # matching the default target ring height. If you change target_height,
        # bump hover_pos[2] by the same amount.
        self.hover_pos = np.asarray(hover_pos, dtype=float)
        self.fire_yaw_thresh = np.deg2rad(fire_yaw_threshold_deg)
        self.fire_tilt_thresh = np.deg2rad(fire_tilt_threshold_deg)
        self.fire_omega_thresh = float(fire_omega_threshold_rps)
        self.fire_pos_thresh = float(fire_pos_threshold_m)
        self.range = float(action_pos_range_m)

    def predict(self, obs):
        d = _decode_obs(obs)

        # Inverse of env's action -> setpoint mapping:
        #   pos_des_x = action[0] * range
        #   pos_des_y = action[1] * range
        #   pos_des_z = (action[2] + 1) * 0.5 * range + 0.5
        #   yaw_des   = action[3] * pi
        #   fire      = action[4] > 0
        a = np.zeros(5, dtype=np.float32)
        a[0] = np.clip(self.hover_pos[0] / self.range, -1.0, 1.0)
        a[1] = np.clip(self.hover_pos[1] / self.range, -1.0, 1.0)
        a[2] = np.clip(2.0 * (self.hover_pos[2] - 0.5) / self.range - 1.0, -1.0, 1.0)

        # Yaw to face the target (relative position is in world frame).
        target_yaw = float(np.arctan2(d["target_rel"][1], d["target_rel"][0]))
        a[3] = np.clip(target_yaw / np.pi, -1.0, 1.0)

        # Fire only when ALL of the following hold:
        #   (a) heading is on target
        #   (b) drone is roughly level
        #   (c) drone isn't rotating fast
        #   (d) drone has actually arrived at the hover position
        # Without (b)/(c)/(d) the agent dumps rounds while still drifting from
        # the previous shot's recoil, and they all miss.
        roll, pitch, current_yaw = quat_to_euler(d["quat"])
        yaw_err = wrap_pi(target_yaw - float(current_yaw))
        omega_norm = float(np.linalg.norm(d["omega_body"]))
        pos_err = float(np.linalg.norm(d["pos"] - self.hover_pos))
        aligned = (abs(yaw_err) < self.fire_yaw_thresh
                   and abs(roll) < self.fire_tilt_thresh
                   and abs(pitch) < self.fire_tilt_thresh
                   and omega_norm < self.fire_omega_thresh
                   and pos_err < self.fire_pos_thresh)
        a[4] = 1.0 if aligned else -1.0
        return a


def run_episode(env, agent, render: bool = False, print_every: int = 20):
    """Run one full episode, print rolling reward, return per-step rewards."""
    obs, info = env.reset()
    rewards = []
    cum = 0.0
    if render:
        env.render()

    print(f"{'step':>5}  {'reward':>8}  {'cum':>10}  {'hits':>5}  {'ammo':>4}")
    print("-" * 44)
    while True:
        action = agent.predict(obs)
        obs, r, terminated, truncated, info = env.step(action)
        rewards.append(float(r))
        cum += r
        if render:
            env.render()

        step = len(rewards)
        if step % print_every == 0 or terminated or truncated:
            print(f"{step:5d}  {r:+8.2f}  {cum:+10.2f}  "
                  f"{info['cumulative_hits']:5d}  {info['ammo']:4d}")

        if terminated or truncated:
            print("-" * 44)
            print(f"end: terminated={terminated} truncated={truncated} "
                  f"final cumulative reward = {cum:+.2f}")
            break

    return np.asarray(rewards, dtype=float)


def plot_rewards(rewards: np.ndarray, save_path: str = "pid_agent_reward.png"):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(rewards, linewidth=0.8)
    ax[0].set_ylabel("reward / step")
    ax[0].grid(True, alpha=0.3)
    ax[0].axhline(0.0, color="black", linewidth=0.5)

    ax[1].plot(np.cumsum(rewards), color="C1")
    ax[1].set_xlabel("env step")
    ax[1].set_ylabel("cumulative reward")
    ax[1].grid(True, alpha=0.3)
    ax[1].axhline(0.0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"plot saved to {save_path}")


def main():
    p = argparse.ArgumentParser(description="PID-style baseline agent for SimpleRewardEnv")
    p.add_argument("--gun", default="m4_carbine")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--render", action="store_true",
                   help="open the MuJoCo viewer to watch the run")
    p.add_argument("--hover-x", type=float, default=0.0)
    p.add_argument("--hover-y", type=float, default=0.0)
    p.add_argument("--hover-z", type=float, default=1.55)
    p.add_argument("--target-radius", type=float, default=5.0)
    p.add_argument("--fire-thresh-deg", type=float, default=5.0)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    if args.no_plot:
        matplotlib.use("Agg")           # no display

    env = SimpleRewardEnv(
        gun=args.gun,
        n_targets=1,
        target_radius=args.target_radius,
        target_height=1.5,
        seed=args.seed,
        max_episode_steps=args.max_steps,
        render_mode="human" if args.render else None,
    )
    agent = PIDAgent(
        hover_pos=(args.hover_x, args.hover_y, args.hover_z),
        fire_yaw_threshold_deg=args.fire_thresh_deg,
        action_pos_range_m=env.action_pos_range_m,
    )

    rewards = run_episode(env, agent, render=args.render)
    env.close()

    if not args.no_plot:
        plot_rewards(rewards)


if __name__ == "__main__":
    main()
