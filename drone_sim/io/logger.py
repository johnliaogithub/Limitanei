"""Trajectory and event recorder.

A `TrajectoryRecorder` buffers two streams during a run:

  1. Per-step PHYSICS STATE (qpos, qvel, ctrl, plus any extras you pass in):
     dense, regular timeseries you'll want to plot.

  2. Discrete EVENTS (shots, hits, resets): sparse, time-stamped records
     stored as JSON-serializable dicts.

On `save()` everything is written into a single compressed `.npz` along
with the model XML and the run's config — so the replayer doesn't need to
share *any* code with the simulator that produced the file.

File layout
-----------
    trajectory.npz
        t              (N,)            sim time per frame
        qpos           (N, model.nq)   full pose vector  (drone + casings + targets)
        qvel           (N, model.nv)   full velocity vector
        ctrl           (N, 4)          per-rotor thrust commands
        pos_des        (N, 3)          position setpoint (controller input)
        yaw_des        (N,)            yaw setpoint
        wind           (N, 3)          world-frame wind at this step
        ammo           (N,)            rounds remaining in the magazine
        xfrc_drone     (N, 6)          recoil + drag wrench applied to drone
        events         scalar str      JSON-encoded list of event dicts
        metadata       scalar str      JSON-encoded dict (XML, config, args)

Stream `qpos` + `qvel` is enough to *exactly* reproduce the visual state
on replay (we just write them back and call mj_forward — no physics
needed). Events let us re-paint target colors and print a hit log.
"""
import json
from collections import defaultdict
from typing import Optional

import numpy as np


class TrajectoryRecorder:
    def __init__(self, path: str, decimation: int = 5):
        """`decimation` = record every Nth physics step. With our 500 Hz
        sim, decimation=5 gives 100 Hz logs (cheap, plenty for analysis)."""
        self.path = path
        self.decimation = max(1, int(decimation))
        self._step = 0
        self._buffers = defaultdict(list)
        self._events = []

    # ---- per-step state ----------------------------------------------------
    def record(self, t: float, data, **extras):
        """Call once per physics step. `data` is the live `mujoco.MjData`.
        Pass any other quantities you want to plot as kwargs."""
        self._step += 1
        if self._step % self.decimation != 0:
            return
        self._buffers["t"].append(float(t))
        self._buffers["qpos"].append(np.array(data.qpos, copy=True))
        self._buffers["qvel"].append(np.array(data.qvel, copy=True))
        self._buffers["ctrl"].append(np.array(data.ctrl, copy=True))
        for k, v in extras.items():
            self._buffers[k].append(_to_array(v))

    # ---- discrete events ---------------------------------------------------
    def event(self, t: float, kind: str, **fields):
        """Log a discrete event (shot fired, target hit, prop hit, reset...)."""
        e = {"t": float(t), "kind": str(kind)}
        for k, v in fields.items():
            e[k] = _to_jsonable(v)
        self._events.append(e)

    # ---- save --------------------------------------------------------------
    def save(self, metadata: Optional[dict] = None):
        if not self._buffers["t"]:
            print(f"[recorder] nothing recorded; skipping write to {self.path}")
            return
        arrays = {k: np.asarray(v) for k, v in self._buffers.items()}
        np.savez_compressed(
            self.path,
            metadata=np.array(json.dumps(metadata or {})),
            events=np.array(json.dumps(self._events)),
            **arrays,
        )
        n = len(self._buffers["t"])
        print(f"[recorder] saved {n} frames + {len(self._events)} events "
              f"to {self.path}")


# ---------------------------------------------------------------------------
#  small helpers to coerce things into something np / JSON can swallow
# ---------------------------------------------------------------------------
def _to_array(v):
    if isinstance(v, np.ndarray):
        return v.copy()
    if np.isscalar(v):
        return np.asarray(v)
    return np.asarray(v)


def _to_jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_jsonable(x) for k, x in v.items()}
    return v


# ---------------------------------------------------------------------------
#  Loader (used by replay.py and any analysis script)
# ---------------------------------------------------------------------------
def load_trajectory(path: str):
    """Load a recorded run. Returns (arrays_dict, metadata_dict, events_list)."""
    arch = np.load(path, allow_pickle=False)
    meta = json.loads(str(arch["metadata"].item()))
    events = json.loads(str(arch["events"].item()))
    arrays = {k: arch[k] for k in arch.files if k not in ("metadata", "events")}
    return arrays, meta, events
