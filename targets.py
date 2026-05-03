"""Static targets in the world, with ray-cast hit detection.

Each target is a sphere geom that doesn't physically collide with anything
(contype=16, conaffinity=0) - it's hit-tested by `mj_ray` from the muzzle
along the firing direction at the moment each shot is fired. Targets
accumulate hits and fade from white toward red as they take damage.

Why ray-cast instead of letting bullets collide naturally?
----------------------------------------------------------
Bullets travel at 360-940 m/s. With a 2 ms physics timestep that's 0.7-1.9 m
of travel per step, which would tunnel through any reasonably-sized target
geom. Even MuJoCo's continuous collision detection breaks down at those
speeds. At realistic engagement ranges (5-50 m), a bullet remains within
~95 % of its muzzle velocity, so the trajectory drop is tiny and a straight
ray cast at firing time is *more* accurate than trying to integrate a
fast-moving rigid body through the contact solver.
"""
from typing import List, Tuple, Optional
import numpy as np
import mujoco


# ---------------------------------------------------------------------------
#  Layout helpers
# ---------------------------------------------------------------------------

def default_ring(n: int = 5, radius: float = 5.0, height: float = 1.5,
                 size: float = 0.20) -> List[Tuple[Tuple[float, float, float], float]]:
    """A ring of `n` targets around the origin at the given radius and height."""
    out = []
    for i in range(n):
        theta = 2.0 * np.pi * i / n
        out.append(((radius * np.cos(theta), radius * np.sin(theta), height), size))
    return out


def target_xml(targets_data) -> str:
    """XML for the target geoms. They're contype=16 with conaffinity=0,
    meaning they don't collide with anything physically — only ray-cast can
    "see" them. This avoids spurious contact forces if a casing or the drone
    drifts into one."""
    chunks = []
    for i, (pos, size) in enumerate(targets_data):
        x, y, z = pos
        chunks.append(f"""
    <body name="target_{i}_body" pos="{x} {y} {z}">
      <geom name="target_{i}" type="sphere" size="{size}"
            rgba="1.0 1.0 1.0 1.0" contype="16" conaffinity="0"/>
    </body>""")
    return "".join(chunks)


# ---------------------------------------------------------------------------
#  Target objects + the field manager
# ---------------------------------------------------------------------------

class Target:
    def __init__(self, target_id: int, geom_id: int,
                 position, size: float, max_hits: int = 10):
        self.id = target_id
        self.geom_id = geom_id
        self.pos = np.asarray(position, dtype=float)
        self.size = float(size)
        self.max_hits = int(max_hits)
        self.hits = 0


class TargetField:
    """Owns the targets, casts rays for hit detection, paints damage."""

    def __init__(self, model, data, targets_data, drone_body_id: int,
                 max_range: float = 200.0):
        self.model = model
        self.data = data
        self.drone_body_id = int(drone_body_id)
        self.max_range = float(max_range)
        self.targets: List[Target] = []
        for i, (pos, size) in enumerate(targets_data):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"target_{i}")
            if gid >= 0:
                self.targets.append(Target(i, gid, pos, size))
        self._geom_to_target = {t.geom_id: t for t in self.targets}

    # -------- ray-cast hit detection ---------------------------------------
    def cast_ray(self, origin, direction) -> Tuple[Optional[Target], Optional[np.ndarray]]:
        """Fire one ray. Returns (target, hit_pos) on hit, else (None, None)."""
        d = np.asarray(direction, dtype=float)
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            return None, None
        d = d / n

        geomid_out = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(
            self.model, self.data,
            np.asarray(origin, dtype=float), d,
            None,                   # geomgroup filter (None = all groups)
            1,                      # include static geoms (floor counts as static)
            self.drone_body_id,     # exclude the drone (don't hit our own gun barrel)
            geomid_out,
        )
        if dist < 0.0 or dist > self.max_range:
            return None, None
        gid = int(geomid_out[0])
        if gid not in self._geom_to_target:
            return None, None
        t = self._geom_to_target[gid]
        t.hits += 1
        self._update_color(t)
        hit_pos = np.asarray(origin, dtype=float) + dist * d
        return t, hit_pos

    def cast_shot(self, origin, fire_dir, n_pellets: int,
                  spread_deg: float, rng) -> List[Tuple[Target, np.ndarray]]:
        """Cast `n_pellets` rays from a single shot (1 for rifles/SMGs, 9 for
        00-buckshot). Each pellet gets independent Gaussian angular jitter
        with std-dev `spread_deg`. Returns the list of (target, hit_pos)
        pairs that landed."""
        hits = []
        fd = np.asarray(fire_dir, float)
        fd = fd / np.linalg.norm(fd)
        sigma = np.radians(max(0.0, spread_deg))
        # two perpendicular axes for the spread cone
        helper = np.array([0.0, 0.0, 1.0]) if abs(fd[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(fd, helper); e1 /= np.linalg.norm(e1)
        e2 = np.cross(fd, e1)
        for _ in range(max(1, n_pellets)):
            if sigma > 0.0:
                d = fd + sigma * float(rng.standard_normal()) * e1 \
                       + sigma * float(rng.standard_normal()) * e2
                d /= np.linalg.norm(d)
            else:
                d = fd
            t, hp = self.cast_ray(origin, d)
            if t is not None:
                hits.append((t, hp))
        return hits

    # -------- visual feedback ----------------------------------------------
    def _update_color(self, t: Target):
        """White when fresh -> bright red when destroyed."""
        frac = min(1.0, t.hits / max(1, t.max_hits))
        self.model.geom_rgba[t.geom_id, 0] = 1.0
        self.model.geom_rgba[t.geom_id, 1] = max(0.0, 1.0 - frac)
        self.model.geom_rgba[t.geom_id, 2] = max(0.0, 1.0 - frac)

    # -------- nearest-target query (used by the yaw aim setpoint) ----------
    def nearest(self, drone_pos) -> Optional[Target]:
        if not self.targets:
            return None
        dp = np.asarray(drone_pos, float)
        best = None
        best_d = float("inf")
        for t in self.targets:
            d = float(np.linalg.norm(t.pos - dp))
            if d < best_d:
                best_d = d
                best = t
        return best

    def reset(self):
        for t in self.targets:
            t.hits = 0
            self.model.geom_rgba[t.geom_id, :] = (1.0, 1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
#  Aim helpers
# ---------------------------------------------------------------------------

def yaw_to_aim(drone_pos, target_pos) -> float:
    """The yaw angle (rad) such that body +x points horizontally at the
    target. Pitch isn't included — see the README for why pitching to aim
    while hovering is fundamentally different from yawing to aim."""
    dx = target_pos[0] - drone_pos[0]
    dy = target_pos[1] - drone_pos[1]
    return float(np.arctan2(dy, dx))
