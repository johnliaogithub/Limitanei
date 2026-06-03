#!/usr/bin/env python3
"""Evaluate a trained PPO model on the drone RL environment.

Examples
--------
# Live MuJoCo viewer, 30 m target:
python eval.py experiments/experiment1/models/ppo_e1_30m_hk416.zip

# Slower playback and save trajectory plot:
python eval.py experiments/experiment1/models/ppo_e1_30m_hk416.zip --speed 0.5 --save-plot out.png

# Headless (no viewer) — just stats and plot:
python eval.py experiments/experiment1/models/ppo_e1_30m_hk416.zip --headless

# Save an MP4 (requires opencv-python):
python eval.py experiments/experiment1/models/ppo_e1_30m_hk416.zip --save-video flight.mp4

# Hover-only model (ZeroTargetEnv):
python eval.py experiments/experiment0/ppo_e0_thrust.zip --env zero
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from drone_sim.rl.custom_envs import ZeroTargetEnv, SingleTargetEnv
from drone_sim.rl.networks import SplitExtractor
from drone_sim.rl.viz import visualize_episode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained PPO model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("model", help="Path to .zip model checkpoint")
    p.add_argument(
        "--env",
        choices=["zero", "single"],
        default="single",
        help="Environment: 'zero' = hover only, 'single' = one stationary target",
    )
    p.add_argument(
        "--target-radius",
        type=float,
        default=30.0,
        metavar="M",
        help="Target spawn distance in metres",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--steps",
        type=int,
        default=500,
        metavar="N",
        help="Max episode steps",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        metavar="X",
        help="Playback speed multiplier (0 = as fast as possible)",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="Skip MuJoCo viewer — print stats and save plot only",
    )
    p.add_argument(
        "--save-video",
        metavar="PATH",
        help="Save rendered episode to MP4 (requires opencv-python)",
    )
    p.add_argument(
        "--save-plot",
        metavar="PATH",
        default="trajectory.png",
        help="Path for the trajectory/reward PNG",
    )
    return p.parse_args()


def _make_env(env_cls, target_radius: float, seed: int, steps: int, render: bool):
    kwargs = dict(
        control_level="thrust",
        target_radius=target_radius,
        action_pos_range_m=target_radius + 10.0,
        seed=seed,
        max_episode_steps=steps,
    )
    if render:
        kwargs["render_mode"] = "human"
    return env_cls(**kwargs)


def run_headless(model, env_cls, target_radius: float, seed: int, steps: int, save_plot: str):
    """Run one episode without a viewer; print a per-step table and save a reward plot."""
    env = _make_env(env_cls, target_radius, seed, steps, render=False)
    obs, _ = env.reset(seed=seed)

    rewards, hits_per_step, positions = [], [], []
    cum = 0.0

    print(f"{'step':>5}  {'reward':>8}  {'cum_reward':>11}  {'total_hits':>10}")
    print("-" * 44)

    while True:
        action = model.predict(obs, deterministic=True)[0]
        obs, r, terminated, truncated, info = env.step(action)

        rewards.append(float(r))
        hits_per_step.append(info.get("hits", 0))
        positions.append(env.unwrapped.data.qpos[:3].copy())
        cum += r
        step = len(rewards)

        if step % 50 == 0 or terminated or truncated:
            print(f"{step:5d}  {r:+8.2f}  {cum:+11.2f}  {sum(hits_per_step):10d}")

        if terminated or truncated:
            reason = "crashed" if terminated else "truncated"
            print("-" * 44)
            print(
                f"End ({reason}) at step {step}  |  "
                f"reward {cum:+.1f}  |  hits {sum(hits_per_step)}"
            )
            break

    env.close()

    rewards = np.array(rewards)
    _plot_rewards(rewards, save_plot)
    return np.array(positions), rewards


def _plot_rewards(rewards: np.ndarray, save_plot: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(rewards, linewidth=0.8, color="steelblue")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("Reward / step")
    ax1.set_title("Per-step reward")
    ax1.grid(True, alpha=0.3)

    ax2.plot(np.cumsum(rewards), color="C1")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Cumulative reward")
    ax2.set_title("Cumulative reward")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_plot, dpi=120)
    print(f"Plot saved → {save_plot}")
    plt.show()


def main():
    args = parse_args()
    env_cls = ZeroTargetEnv if args.env == "zero" else SingleTargetEnv

    # Load model against a throw-away env so SB3 rebuilds the policy correctly.
    dummy = _make_env(env_cls, args.target_radius, args.seed, args.steps, render=False)
    model = PPO.load(args.model, env=dummy)
    dummy.close()

    if args.headless:
        run_headless(
            model, env_cls,
            target_radius=args.target_radius,
            seed=args.seed,
            steps=args.steps,
            save_plot=args.save_plot,
        )
    else:
        visualize_episode(
            model,
            env_cls=env_cls,
            target_radius=args.target_radius,
            seed=args.seed,
            max_steps=args.steps,
            speed=args.speed,
            save_path=args.save_plot,
            save_video=args.save_video,
        )


if __name__ == "__main__":
    main()
