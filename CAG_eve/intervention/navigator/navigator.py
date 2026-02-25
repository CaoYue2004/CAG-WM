# eve/navigation/navigator.py
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...util import EveObject


# ============================================================
# Navigator：抽象基类（继承 EveObject）
# ============================================================

class Navigator(EveObject, ABC):
    """
    Navigator：导航器抽象基类
    - reset(): 根据 target 构建导航表（dist_to_target / next_hop 等）
    - step(): 根据 tip pose 生成 route（世界坐标/局部坐标）以及 dist 标量等
    """

    # --- 由子类在 reset/step 中维护的状态 ---
    target_point_world: np.ndarray              # (3,)
    dist_to_target_scalar: float                # 当前 tip 到 target 的最短路距离（标量）
    route_pts_world: np.ndarray                 # (K,3) 世界坐标路径点
    route_pts_local: np.ndarray                 # (K,3) tip 局部坐标路径点（可选）
    k_route: int                                # 输出路径点数量

    @abstractmethod
    def reset(self) -> None:
        """
        target_point_world: 目标点（tracking/world 坐标系）(3,)
        seed: 可选，方便可复现（例如随机化某些策略）
        """
        ...

    @abstractmethod
    def step(
        self,
    ) -> None:
        """
        tip_point_world: tip 位置（tracking/world 坐标）(3,)
        tip_R_world: tip 坐标系到 world 的旋转矩阵 (3,3)
          - 若为 None，子类可只更新 route_pts_world/dist，不必生成 local
        """
        ...

    # --- reset 快照 ---
    def get_reset_state(self) -> Dict[str, Any]:
        state = {
            "target_point_world": getattr(self, "target_point_world", None),
            "k_route": getattr(self, "k_route", None),
        }
        return deepcopy(state)

    # --- step 快照 ---
    def get_step_state(self) -> Dict[str, Any]:
        state = {
            "dist_to_target_scalar": getattr(self, "dist_to_target_scalar", None),
            "route_pts_world": getattr(self, "route_pts_world", None),
            "route_pts_local": getattr(self, "route_pts_local", None),
        }
        return deepcopy(state)