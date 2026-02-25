from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from ..util import EveObject
from ..intervention import Intervention
from ..util.coordtransform import tracking3d_to_2d


# ----------------------------------------------------------------------
# InterimTarget：中间目标（航点/路标）的抽象基类
# - all_coordinates3d：一串中间目标点（tracking 坐标系）
# - reached：是否到达“当前航点”（通常是 all_coordinates3d[0]）
# - threshold：到达判定阈值（距离阈值或投影阈值，由子类定义）
# - intervention：提供环境状态（tracking、目标、血管树等）
# ----------------------------------------------------------------------
class InterimTarget(EveObject, ABC):
    # Needs to be set by implementing classes in step() or reset().
    # Coordinates are in the tracking coordinate space
    all_coordinates3d: List[np.ndarray]
    reached: bool
    threshold: float
    intervention: Intervention

    # ------------------------------------------------------------------
    # 当前中间目标点（通常取队列第一个）
    # 若没有任何中间目标，则返回 None
    # ------------------------------------------------------------------
    @property
    def coordinates3d(self) -> np.ndarray:
        return self.all_coordinates3d[0] if len(self.all_coordinates3d) > 0 else None

    # ------------------------------------------------------------------
    # 当前中间目标的 2D 坐标（由 tracking3d_to_2d 投影得到）
    # 若当前 3D 目标不存在，则返回 None
    # ------------------------------------------------------------------
    @property
    def coordinates2d(self) -> np.ndarray:
        if self.coordinates3d is None:
            return None
        return tracking3d_to_2d(self.coordinates3d)

    # ------------------------------------------------------------------
    # 所有中间目标点的 2D 坐标列表
    # 若没有中间目标点，则返回空列表
    # ------------------------------------------------------------------
    @property
    def all_coordinates2d(self) -> List[np.ndarray]:
        if not self.all_coordinates3d:
            return []
        return tracking3d_to_2d(self.all_coordinates3d)

    @abstractmethod
    def reset(self, episode_nr: int = 0, seed: Optional[int] = None) -> None:
        ...

    @abstractmethod
    def step(self) -> None:
        ...
