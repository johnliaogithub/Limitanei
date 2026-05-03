"""Bullets: pure-Python ballistics + tracer rendering in the viewer.

Bullets travel at 360-940 m/s. With our 2 ms timestep that's 0.7-1.9 m of
travel per step, which would tunnel through any reasonable target geom in
the MuJoCo collision solver — even with continuous collision detection on.
So instead we simulate bullets entirely in Python with explicit ballistics:

    F_drag  =  -0.5 * rho * Cd * A * |v| * v        (per kg of bullet)
    acc     =  g + F_drag / m

The frontal area `A` is approximated from the bullet's mass assuming
lead density (11.34 g/cm^3) and a sphere — close enough for visualization;
for actual external-ballistics accuracy you'd plug in cartridge-specific
G7 coefficients. Each active bullet renders as a short line segment between
its previous and current position in `viewer.user_scn`, giving a tracer
effect.

(Hit-testing against targets uses ray casting — see targets.py — which is
*more* accurate than the integrated bullet position would be at these
speeds.)
"""
import numpy as np
import mujoco


def bullet_drag_const(weapon, rho: float = 1.225, cd: float = 0.30) -> float:
    """Pre-computed multiplier for `acc_drag = -drag_const * |v| * v`."""
    r = (3.0 * weapon.bullet_mass_kg / (4.0 * np.pi * 11340.0)) ** (1.0 / 3.0)
    A = np.pi * r * r
    return 0.5 * rho * cd * A / max(weapon.bullet_mass_kg, 1e-6)


class BulletPool:
    """Simple ballistic projectiles. Position-Verlet integration with
    quadratic air drag. Rendered as segments from previous to current position
    so each bullet leaves a visible streak in the viewer."""

    def __init__(self, pool_size: int, lifetime: float, gravity: float,
                 drag_const: float, rng):
        self.n = pool_size
        self.lifetime = lifetime
        self.g = gravity
        self.drag_const = drag_const
        self.rng = rng

        self.pos = np.zeros((pool_size, 3))
        self.prev_pos = np.zeros((pool_size, 3))
        self.vel = np.zeros((pool_size, 3))
        self.life = np.zeros(pool_size)
        self.next = 0

    def spawn(self, pos: np.ndarray, vel: np.ndarray):
        i = self.next
        self.next = (self.next + 1) % self.n
        self.pos[i] = pos
        self.prev_pos[i] = pos
        self.vel[i] = vel
        self.life[i] = self.lifetime

    def step(self, dt: float):
        active = self.life > 0
        if not np.any(active):
            return
        v = self.vel[active]
        speed = np.linalg.norm(v, axis=1, keepdims=True)
        # acceleration = gravity - drag_const * |v| * v
        acc = np.array([0.0, 0.0, -self.g])[None, :] - self.drag_const * speed * v
        self.prev_pos[active] = self.pos[active]
        self.pos[active] += v * dt + 0.5 * acc * dt * dt
        self.vel[active] += acc * dt
        self.life[active] -= dt
        # cull anything that hit the ground
        below = (self.pos[:, 2] < 0.0) & (self.life > 0)
        self.life[below] = 0.0

    def render(self, user_scn):
        """Add a line geom for each active bullet to the viewer scene.
        user_scn must be reset (ngeom=0) before this call."""
        size = np.array([0.003, 0.0, 0.0])
        rgba = np.array([1.0, 0.85, 0.2, 1.0])
        for i in range(self.n):
            if self.life[i] <= 0:
                continue
            if user_scn.ngeom >= user_scn.maxgeom:
                break
            g = user_scn.geoms[user_scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, size,
                                np.zeros(3), np.zeros(9), rgba)
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 0.005,
                                 self.prev_pos[i], self.pos[i])
            user_scn.ngeom += 1
