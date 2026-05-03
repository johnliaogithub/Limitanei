"""Replay a recorded trajectory in the MuJoCo viewer.

    python replay.py path/to/flight.npz
    python replay.py flight.npz --speed 0.25      # quarter-speed
    python replay.py flight.npz --speed 4.0       # 4x fast-forward

The replayer writes the recorded qpos/qvel back into MuJoCo and calls
mj_forward (kinematic playback — NO physics is run). Target colors are
repainted by walking the events list, so you see hits accumulate at the
right wall-clock time.

Bullets aren't in qpos (they were tracked in Python during the live run),
so they don't appear in replay — but every bullet's spawn was logged as
an event, so a hit summary is printed at the end. Casings DO appear,
because they're real MuJoCo bodies.
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from drone_sim.io.logger import load_trajectory


def replay(path: str, speed: float = 1.0):
    arrays, meta, events = load_trajectory(path)
    if "model_xml" not in meta:
        raise RuntimeError("trajectory file has no model_xml in metadata; "
                           "can't reconstruct scene")
    model = mujoco.MjModel.from_xml_string(meta["model_xml"])
    data = mujoco.MjData(model)

    times = arrays["t"]
    qpos = arrays["qpos"]
    qvel = arrays["qvel"]

    # --- index events for incremental color updates --------------------------
    target_hits_log = {}            # tid -> sorted list of (event_time, hits_after)
    for e in events:
        if e["kind"] == "target_hit":
            tid = int(e["target_id"])
            target_hits_log.setdefault(tid, []).append(
                (float(e["t"]), int(e["hits"]))
            )
    target_geom_id = {}
    for tid in target_hits_log:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"target_{tid}")
        if gid >= 0:
            target_geom_id[tid] = gid

    print(f"[replay] {len(times)} frames covering "
          f"{times[0]:.2f} - {times[-1]:.2f} s "
          f"({(times[-1]-times[0]):.1f}s sim time)")
    print(f"[replay] {len(events)} events  |  speed = {speed}x")
    n_shots = sum(1 for e in events if e["kind"] == "shot")
    n_hits = sum(1 for e in events if e["kind"] == "target_hit")
    n_props = sum(1 for e in events if e["kind"] == "prop_hit")
    print(f"[replay] {n_shots} shots, {n_hits} target hits, {n_props} prop hits")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        wall_t0 = time.time()
        sim_t0 = float(times[0])
        for i in range(len(times)):
            if not viewer.is_running():
                break
            # 1. set physics state
            data.qpos[:] = qpos[i]
            data.qvel[:] = qvel[i]
            mujoco.mj_forward(model, data)

            # 2. paint targets according to events with t <= now
            now = float(times[i])
            for tid, log in target_hits_log.items():
                cur = 0
                for et, hits in log:
                    if et <= now:
                        cur = hits
                    else:
                        break
                gid = target_geom_id.get(tid)
                if gid is not None:
                    frac = min(1.0, cur / 10.0)
                    model.geom_rgba[gid, 1] = max(0.0, 1.0 - frac)
                    model.geom_rgba[gid, 2] = max(0.0, 1.0 - frac)

            viewer.sync()

            # 3. pace to wall clock so 1 sim-second = 1 / speed wall-second
            target_wall = wall_t0 + (now - sim_t0) / max(speed, 1e-6)
            slack = target_wall - time.time()
            if slack > 0:
                time.sleep(slack)


def main():
    p = argparse.ArgumentParser(description="Replay a recorded trajectory")
    p.add_argument("path", help="path to .npz produced by --record")
    p.add_argument("--speed", type=float, default=1.0,
                   help="playback speed factor (1.0 = real-time, 0.25 = slow-mo)")
    args = p.parse_args()
    replay(args.path, args.speed)


if __name__ == "__main__":
    main()
