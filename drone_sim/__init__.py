"""drone_sim: MuJoCo quadrotor + machine-gun simulation.

Public API — import directly from `drone_sim`:

    from drone_sim import DRONE, CTRL, SIM, GUN, DIST, PROJ
    from drone_sim import CascadedController, KeyboardSetpoint, AutonomousSetpoint
    from drone_sim import DroneEnv, SimpleRewardEnv

Submodules also exposed by name:
    drone_sim.physics   -- gun, bullets, casings, disturbances, targets
    drone_sim.control   -- controller, modes
    drone_sim.io        -- logger, replay
    drone_sim.rl        -- env (DroneEnv), custom_envs, agents
"""
from drone_sim.config import (
    DRONE, CTRL, SIM, GUN, DIST, PROJ,
    DroneParams, ControlParams, SimParams,
    GunParams, DisturbanceParams, ProjectileParams,
)
from drone_sim.control.controller import (
    CascadedController, quat_to_euler, wrap_pi,
)
from drone_sim.control.modes import KeyboardSetpoint, AutonomousSetpoint
from drone_sim.rl.env import DroneEnv
from drone_sim.rl.custom_envs import SimpleRewardEnv

__all__ = [
    # config singletons + dataclasses
    "DRONE", "CTRL", "SIM", "GUN", "DIST", "PROJ",
    "DroneParams", "ControlParams", "SimParams",
    "GunParams", "DisturbanceParams", "ProjectileParams",
    # control
    "CascadedController", "quat_to_euler", "wrap_pi",
    "KeyboardSetpoint", "AutonomousSetpoint",
    # gym envs
    "DroneEnv", "SimpleRewardEnv",
]
