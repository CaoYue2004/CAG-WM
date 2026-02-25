from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Optional
import numpy as np

from ...util import EveObject
from ..fluoroscopy import Fluoroscopy


# ----------------------------------------------------------------------
# Target：目标点抽象基类
# - coordinates3d / coordinates2d：目标在 tracking 坐标系下的位置（3D/2D）
# - reached：是否到达
# - threshold：到达判定阈值（距离 < threshold）
# - fluoroscopy：用于读取当前器械/尖端 tracking 的透视对象
# ----------------------------------------------------------------------
class Target(EveObject, ABC):
    # Needs to be set by implementing classes in step() or reset().
    # Coordinates are in the tracking coordinate space
    coordinates3d: np.ndarray
    coordinates2d: np.ndarray
    reached: bool
    threshold: float
    fluoroscopy: Fluoroscopy

    # ------------------------------------------------------------------
    # 抽象 reset：子类必须实现（负责设置目标坐标、阈值、reached 初值等）
    # episode_nr：第几局（用于切换病例/目标）
    # seed：随机种子（用于可复现的随机目标）
    # ------------------------------------------------------------------
    @abstractmethod
    def reset(self, episode_nr: int = 0, seed: Optional[int] = None) -> None:
        ...

    # ------------------------------------------------------------------
    # step：每步更新 reached 状态
    # 默认逻辑：取 fluoroscopy.tracking3d 的第 0 个点当作当前位置
    # ------------------------------------------------------------------
    def step(self) -> None:
        position = self.fluoroscopy.tracking3d[0]
        position_to_target = self.coordinates3d - position

        self.reached = (
            True if np.linalg.norm(position_to_target) < self.threshold else False
        )

    # ------------------------------------------------------------------
    # reset 时的状态快照：返回目标的完整关键状态
    # ------------------------------------------------------------------
    def get_reset_state(self) -> Dict[str, Any]:
        state = {
            "coordinates3d": self.coordinates3d,
            "coordinates2d": self.coordinates2d,
            "reached": self.reached,
            "threshold": self.threshold,
        }
        return deepcopy(state)

    # ------------------------------------------------------------------
    # step 时的状态快照：通常只返回动态量（这里仅 reached）
    # ------------------------------------------------------------------
    def get_step_state(self) -> Dict[str, Any]:
        state = {
            "reached": self.reached,
        }
        return deepcopy(state)
