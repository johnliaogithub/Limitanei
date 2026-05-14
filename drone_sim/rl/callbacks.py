"""SB3 training callbacks for drone RL."""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ShotStatsCallback(BaseCallback):
    """Logs shots-fired and hits per episode during PPO training.

    Accumulates shots_fired / hits from info dicts across all parallel envs.
    Every `print_freq` completed episodes it prints a summary line and records
    the stats to the SB3 logger (shows up in TensorBoard under "shots/").
    """

    def __init__(self, print_freq: int = 10, verbose: int = 1):
        super().__init__(verbose)
        self.print_freq = print_freq
        self._ep_shots: np.ndarray | None = None
        self._ep_hits: np.ndarray | None = None
        self._buf_shots: list[float] = []
        self._buf_hits: list[float] = []
        self._n_episodes = 0

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._ep_shots = np.zeros(n)
        self._ep_hits = np.zeros(n)

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]
        for i, (done, info) in enumerate(zip(dones, infos)):
            self._ep_shots[i] += info.get("shots_fired", 0)
            self._ep_hits[i] += info.get("hits", 0)
            if done:
                self._buf_shots.append(float(self._ep_shots[i]))
                self._buf_hits.append(float(self._ep_hits[i]))
                self._ep_shots[i] = 0.0
                self._ep_hits[i] = 0.0
                self._n_episodes += 1

                if self._n_episodes % self.print_freq == 0:
                    n = min(self.print_freq, len(self._buf_shots))
                    mean_shots = np.mean(self._buf_shots[-n:])
                    mean_hits  = np.mean(self._buf_hits[-n:])
                    if self.verbose >= 1:
                        print(
                            f"[ep {self._n_episodes:5d}]  "
                            f"shots/ep={mean_shots:6.1f}  "
                            f"hits/ep={mean_hits:.2f}"
                        )
                    self.logger.record("shots/mean_per_ep",   mean_shots)
                    self.logger.record("shots/mean_hits_per_ep", mean_hits)
        return True
