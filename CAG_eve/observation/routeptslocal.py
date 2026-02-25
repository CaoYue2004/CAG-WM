import numpy as np
import gymnasium as gym

from .observation import Observation
from ..intervention import Intervention
from ..intervention.navigator import GraphNavigator

class RoutePtsLocal(Observation):

    def __init__(
        self,
        intervention: Intervention,
        name: str = "route_pts_local",
        bound: float = 1e4,
    ) -> None:
        self.intervention = intervention
        self.name = name
        self.bound = float(bound)

        K = int(self.intervention.navigator.k_route)
        shape = (K, 3)
        self.obs = np.zeros(shape, dtype=np.float32)

    @property
    def space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=-self.bound,
            high=self.bound,
            shape=self.obs.shape,
            dtype=np.float32,
        )

    def reset(self, episode_nr: int = 0) -> None:
        _ = episode_nr

        route = self.intervention.navigator.route_pts_local.astype(np.float32)
        self.obs = route

    def step(self) -> None:
        route = self.intervention.navigator.route_pts_local.astype(np.float32)
        self.obs = route






