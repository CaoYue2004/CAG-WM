# pylint: disable=unused-argument
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym
from ...util import EveObject
from ..simulation import Simulation


# -------------------------
# Fluoroscopy：透视成像/荧光透视（C-arm）抽象接口
# 负责提供：图像（image） + 2D/3D tracking（血管/目标/器械投影与位置）
# -------------------------
class Fluoroscopy(EveObject):
    # ----------- 配置/参数类字段（类型注解，不是赋值） -----------
    image_frequency: float
    image_rot_zx: Tuple[float, float]
    image_center: Optional[Tuple[float, float, float]]
    field_of_view: Optional[Tuple[float, float]]
    image_space: gym.spaces.Box
    image: np.ndarray
    tracking3d_space: gym.spaces.Box
    tracking3d_space_episode: gym.spaces.Box
    tracking2d_space: gym.spaces.Box
    tracking2d_space_episode: gym.spaces.Box
    tracking3d: np.ndarray
    tracking2d: np.ndarray
    device_trackings3d: Optional[List[np.ndarray]]
    device_trackings2d: Optional[List[np.ndarray]]

    def step(self) -> None:
        ...

    def reset(self, episode_nr: int = 0) -> None:
        ...

    def get_reset_state(self) -> Dict[str, Any]:
        state = {
            "image_frequency": self.image_frequency,
            "image_rot_zx": self.image_rot_zx,
            "image_center": self.image_center,
            "field_of_view": self.field_of_view,
            "image_space": self.image_space,
            "image": self.image,
            "tracking2d_space": self.tracking2d_space,
            "tracking2d_space_episode": self.tracking2d_space_episode,
            "tracking3d_space": self.tracking3d_space,
            "tracking3d_space_episode": self.tracking3d_space_episode,
            "tracking3d": self.tracking3d,
            "tracking2d": self.tracking2d,
            "device_trackings3d": self.device_trackings3d,
            "device_trackings2d": self.device_trackings2d,
        }
        return deepcopy(state)

    def get_step_state(self) -> Dict[str, Any]:
        state = {
            "image_frequency": self.image_frequency,
            "image_rot_zx": self.image_rot_zx,
            "image_center": self.image_center,
            "field_of_view": self.field_of_view,
            "image": self.image,
            "tracking3d": self.tracking3d,
            "tracking2d": self.tracking2d,
            "device_trackings3d": self.device_trackings3d,
            "device_trackings2d": self.device_trackings2d,
        }
        return deepcopy(state)


# -------------------------
# SimulatedFluoroscopy：Fluoroscopy 的一个具体实现分支
# 表示“由仿真 Simulation 驱动生成”的透视系统（而不是读取真实 DICOM/视频）
# -------------------------
class SimulatedFluoroscopy(Fluoroscopy):
    simulation: Simulation
