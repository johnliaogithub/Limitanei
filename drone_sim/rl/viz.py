"""Visualization utilities for drone RL episodes."""
import time
import numpy as np
import matplotlib.pyplot as plt
import mujoco
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from drone_sim.control.controller import quat_to_euler
from drone_sim.sim import quat_to_rot
from drone_sim.config import SIM


_LINE_IDLE   = np.array([1.0, 0.25, 0.25, 0.6], dtype=np.float32)  # dim red
_LINE_FIRING = np.array([1.0, 1.0,  0.1,  1.0], dtype=np.float32)  # bright yellow


def visualize_episode(
    model,
    env_cls=None,
    target_radius: float = 20.0,
    seed: int = 42,
    max_steps: int = 500,
    speed: float = 1.0,
    cam_distance: float | None = None,
    save_path: str = "trajectory.png",
    flash: bool = True,
    **env_kwargs,
):
    """Run and visualize one episode in the MuJoCo viewer.

    Parameters
    ----------
    model         : SB3 model (must have .predict)
    env_cls       : Environment class (default: SingleTargetEnv)
    target_radius : metres
    seed          : RNG seed
    max_steps     : episode length cap
    speed         : playback multiplier — 1.0 = real-time, 2.0 = 2× faster,
                    0 = max speed (no sleep)
    cam_distance  : viewer camera distance; defaults to target_radius * 1.5
    save_path     : where to save the trajectory PNG
    flash         : if True, the aiming line turns bright yellow when the gun
                    fires; if False the line stays dim red at all times
    **env_kwargs  : forwarded to env_cls()
    """
    if env_cls is None:
        from drone_sim.rl.custom_envs import SingleTargetEnv
        env_cls = SingleTargetEnv

    if cam_distance is None:
        cam_distance = target_radius * 1.5

    env = env_cls(
        control_level="thrust",
        render_mode="human",
        target_radius=target_radius,
        action_pos_range_m=target_radius + 10.0,
        seed=seed,
        **env_kwargs,
    )
    obs, info = env.reset(seed=seed)

    target_pos  = obs[:3] + obs[14:17] * target_radius
    drone_start = obs[:3].copy()

    env.render()
    viewer = env.unwrapped._viewer
    if viewer is not None:
        viewer.cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.distance  = float(cam_distance)
        viewer.cam.elevation = -25.0
        viewer.cam.azimuth   = 135.0
        viewer.cam.lookat[:] = (drone_start + target_pos) / 2.0
        viewer.sync()
    else:
        print("Viewer not available — running headless.")

    # Real-time: sleep step_duration/speed seconds per env.step()
    step_duration = env.unwrapped.frame_skip * SIM.timestep  # seconds of sim per step
    sleep_target  = (step_duration / speed) if speed > 0 else 0.0

    fire_dir_body = np.asarray(env.unwrapped.weapon.fire_dir_body, float)
    mount_offset  = np.asarray(env.unwrapped.weapon.mount_offset_m, float)

    positions, quats, rewards_log, hits_log = [], [], [], []

    for step in range(max_steps):
        t0 = time.perf_counter()

        action = model.predict(obs, deterministic=True)[0]
        obs, reward, terminated, truncated, info = env.step(action)

        # aiming line — color flashes bright yellow when firing
        if viewer is not None:
            viewer.user_scn.ngeom = 0
            R_r  = quat_to_rot(obs[3:7])
            gdir = R_r @ fire_dir_body
            muz  = obs[:3] + R_r @ (mount_offset + fire_dir_body * 0.30)

            if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                firing = flash and info["shots_fired"] > 0
                color  = _LINE_FIRING if firing else _LINE_IDLE
                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(
                    g, mujoco.mjtGeom.mjGEOM_LINE,
                    np.zeros(3), np.zeros(3), np.zeros(9), color,
                )
                mujoco.mjv_connector(
                    g, mujoco.mjtGeom.mjGEOM_LINE, 0.004, muz, muz + gdir * 60.0
                )
                viewer.user_scn.ngeom += 1

        env.render()

        positions.append(obs[:3].copy())
        quats.append(obs[3:7].copy())
        rewards_log.append(float(reward))
        hits_log.append(int(info["hits"]))

        if sleep_target > 0:
            slack = sleep_target - (time.perf_counter() - t0)
            if slack > 0:
                time.sleep(slack)

        if terminated or truncated:
            print(
                f"Episode ended at step {step + 1}  "
                f"({'crashed' if terminated else 'truncated'})"
            )
            break

    env.close()

    positions   = np.array(positions)
    rewards_log = np.array(rewards_log)
    total_hits  = sum(hits_log)
    print(
        f"Steps: {len(positions)} / {max_steps}  |  "
        f"Total reward: {rewards_log.sum():.1f}  |  Hits: {total_hits}"
    )

    _plot_episode(
        positions, quats, rewards_log, hits_log,
        target_pos, target_radius, total_hits, save_path,
    )
    return positions, rewards_log


def _plot_episode(positions, quats, rewards_log, hits_log,
                  target_pos, target_radius, total_hits, save_path):
    rolls, pitches, _ = zip(*[quat_to_euler(q) for q in quats])
    rolls   = np.degrees(rolls)
    pitches = np.degrees(pitches)
    t = np.arange(len(positions)) * SIM.timestep * 10  # seconds (frame_skip=10)

    fig = plt.figure(figsize=(17, 9))

    ax3d = fig.add_subplot(2, 3, (1, 4), projection="3d")
    ax3d.plot(positions[:, 0], positions[:, 1], positions[:, 2],
              lw=1.2, color="royalblue", label="drone path")
    ax3d.scatter(*positions[0],  s=80,  c="green",  zorder=5, label="start")
    ax3d.scatter(*positions[-1], s=80,  c="red",    zorder=5, label="end")
    ax3d.scatter(*target_pos,    s=200, c="orange", marker="*", zorder=5, label="target")
    hit_steps = [i for i, h in enumerate(hits_log) if h > 0]
    if hit_steps:
        ax3d.scatter(
            positions[hit_steps, 0], positions[hit_steps, 1], positions[hit_steps, 2],
            s=120, c="yellow", edgecolors="black", zorder=6,
            label=f"hits ({total_hits})",
        )
    ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3-D Trajectory"); ax3d.legend(fontsize=8)

    ax1 = fig.add_subplot(2, 3, 2)
    ax1.plot(t, positions[:, 2], color="royalblue")
    ax1.axhline(target_pos[2], color="orange", ls="--", lw=0.8, label="target z")
    ax1.set_ylabel("Altitude (m)"); ax1.set_title("Altitude"); ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 3, 3)
    ax2.plot(t, rolls,   label="roll")
    ax2.plot(t, pitches, label="pitch")
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("Angle (°)"); ax2.set_title("Roll / Pitch"); ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 3, 5)
    ax3.plot(t, rewards_log, lw=0.8, color="steelblue")
    ax3.axhline(0, color="black", lw=0.5)
    ax3.set_ylabel("Reward"); ax3.set_title("Reward / step")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(2, 3, 6)
    ax4.plot(t, np.cumsum(rewards_log), color="C1")
    ax4.set_ylabel("Cumulative reward"); ax4.set_title("Cumulative reward")
    ax4.grid(True, alpha=0.3)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel("Time (s)")

    plt.suptitle(
        f"target_radius={target_radius} m  |  "
        f"reward={rewards_log.sum():.0f}  hits={total_hits}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.show()
