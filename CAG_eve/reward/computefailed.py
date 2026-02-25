from typing import Optional
from .reward import Reward
from ..intervention import Intervention


class ComputeFailed(Reward):
    def __init__(self, intervention: Intervention, factor: float) -> None:
        self.intervention = intervention
        self.factor = factor
        self._last_error = False

    def step(self) -> None:
        error = bool(self.intervention.simulation.simulation_error)
        # 只在首次出错那一步惩罚一次
        self.reward = -self.factor if (error and not self._last_error) else 0.0
        self._last_error = error

    def reset(self, episode_nr: int = 0) -> None:
        self.reward = 0.0
        self._last_error = False


