"""Shared neural-network building blocks for drone RL policies.

Keeping these in the package (not inlined in notebooks) means every notebook
that saves or loads a PPO model with SplitExtractor can just do:

    from drone_sim.rl.networks import SplitExtractor, FLIGHT_DIM, TARGET_DIM
"""
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Observation layout (must match DroneEnv._get_obs):
#   pos(3) + quat(4) + vel(3) + omega_body(3) + ammo(1) = 14  [flight state]
#   target_rel_normalized(3)                              =  3  [task state]
FLIGHT_DIM = 14
TARGET_DIM = 3


class SplitExtractor(BaseFeaturesExtractor):
    """Two-stream SB3 features extractor.

    flight_enc : obs[0:-3]  → Linear(flight_dim, 64) → Tanh → 64
    target_enc : obs[-3:]   → Linear(3,           16) → Tanh → 16
    Output: concat → features_dim=80

    flight_dim is inferred from obs_space.shape[0] - TARGET_DIM, so this
    works for both the default 17-dim obs (flight_dim=14, obs_pos_xy=True)
    and the 15-dim obs (flight_dim=12, obs_pos_xy=False).

    The split keeps the BC-pretrained hover path (flight_enc) isolated from
    the target-aiming path (target_enc) so PPO can train them independently.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 80):
        super().__init__(observation_space, features_dim)
        flight_dim = observation_space.shape[0] - TARGET_DIM
        self.flight_dim = flight_dim
        self.flight_enc = nn.Sequential(nn.Linear(flight_dim, 64), nn.Tanh())
        self.target_enc = nn.Sequential(nn.Linear(TARGET_DIM, 16), nn.Tanh())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        f = self.flight_enc(obs[:, :self.flight_dim])
        t = self.target_enc(obs[:, self.flight_dim:])
        return torch.cat([f, t], dim=1)
