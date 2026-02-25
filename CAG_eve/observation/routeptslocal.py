import numpy as np
import gymnasium as gym

from .observation import Observation
from ..intervention import Intervention
from ..intervention.navigator import GraphNavigator

class RoutePtsLocal(Observation):
    """
    观测：navigator.route_pts_local
    输出：shape (K,3) 或 flatten 为 (K*3,)
    """

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

        # 关键：确保 reset 后 route 已经可用
        # 你可以选择：
        # A) 只依赖外部 env 在 reset 时已经调用过 navigator.reset() + navigator.step()
        # B) 这里自己再调用一次 navigator.step()（更鲁棒，但要求 tip/target 已ready）
        route = self.intervention.navigator.route_pts_local.astype(np.float32)
        self.obs = route

    def step(self) -> None:
        # 每步先更新 navigator，再取出 route
        route = self.intervention.navigator.route_pts_local.astype(np.float32)
        self.obs = route





