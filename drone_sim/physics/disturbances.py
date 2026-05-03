"""Stochastic disturbances that act on the drone or the gun.

  * WindModel       -> mean wind plus an Ornstein-Uhlenbeck turbulent gust
                       process (band-limited Gaussian noise with a sensible
                       correlation time).
  * recoil_noise    -> per-shot multiplicative impulse jitter + small
                       angular wobble of the firing direction.
  * drag_force      -> aerodynamic drag on the drone in apparent wind.

All of these take a numpy.random.Generator so the simulation is fully
reproducible from a single integer seed.
"""
from typing import Tuple
import numpy as np


class WindModel:
    """Mean wind + Ornstein-Uhlenbeck turbulence in 3D.

    OU process for each component independently:

        dW/dt  =  -W/tau  +  sigma * sqrt(2/tau) * white_noise

    The stationary distribution is N(0, sigma^2); tau sets the time scale
    over which gusts evolve (1-3 s is typical for wind near the ground).
    Discrete update used here:

        W_{k+1}  =  W_k  -  (dt/tau) * W_k  +  sigma * sqrt(2*dt/tau) * z

    where z ~ N(0, 1).
    """

    def __init__(self, mean_mps, sigma_mps, tau_s, rng: np.random.Generator):
        self.mean = np.asarray(mean_mps, float)
        self.sigma = float(sigma_mps)
        self.tau = float(tau_s)
        self.gust = np.zeros(3)
        self.rng = rng

    def step(self, dt: float) -> np.ndarray:
        if self.sigma > 0.0 and self.tau > 0.0:
            decay = dt / self.tau
            kick = self.sigma * np.sqrt(max(0.0, 2.0 * decay))
            self.gust += -decay * self.gust + kick * self.rng.standard_normal(3)
        return self.mean + self.gust


def drag_force(drone_vel_world: np.ndarray, wind_vel_world: np.ndarray,
               cd_A: float, air_density: float = 1.225) -> np.ndarray:
    """Quadratic aerodynamic drag on the airframe in apparent wind.

        F  =  0.5 * rho * Cd * A * |v_app| * v_app

    with apparent wind  v_app = wind - drone_vel  (force opposes motion
    *relative* to the surrounding air). Returned as a world-frame N vector.
    """
    v_app = wind_vel_world - drone_vel_world
    speed = float(np.linalg.norm(v_app))
    if speed < 1e-6:
        return np.zeros(3)
    return 0.5 * air_density * cd_A * speed * v_app


def recoil_noise(rng: np.random.Generator, fire_dir_body: np.ndarray,
                 sigma_impulse: float, sigma_angle_deg: float
                 ) -> Tuple[float, np.ndarray]:
    """Sample a per-shot multiplicative impulse factor and a jittered firing
    direction. Returns (impulse_factor, jittered_dir_unit).

    Real-gun recoil varies from shot to shot due to powder charge and bullet
    weight tolerances, friction in the action, barrel temperature, and the
    fact that the muzzle whips around as the bullet exits. 2-5 % impulse
    sigma and ~0.5-1.5 deg angular sigma are realistic for quality ammo.
    """
    factor = 1.0
    if sigma_impulse > 0.0:
        factor = max(0.0, 1.0 + sigma_impulse * float(rng.standard_normal()))

    fd = np.asarray(fire_dir_body, float)
    fd = fd / np.linalg.norm(fd)
    if sigma_angle_deg > 0.0:
        sigma_rad = np.radians(sigma_angle_deg)
        # build two axes perpendicular to fd
        helper = np.array([0.0, 0.0, 1.0]) if abs(fd[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(fd, helper); e1 /= np.linalg.norm(e1)
        e2 = np.cross(fd, e1)
        a = sigma_rad * float(rng.standard_normal())
        b = sigma_rad * float(rng.standard_normal())
        fd = fd + a * e1 + b * e2
        fd = fd / np.linalg.norm(fd)
    return factor, fd
