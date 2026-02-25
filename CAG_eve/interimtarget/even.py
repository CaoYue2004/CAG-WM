from typing import Optional
import numpy as np

from .interimtarget import InterimTarget
from ..pathfinder import Pathfinder
from ..intervention import Intervention


# ----------------------------------------------------------------------
# Even：一种“等距中间目标”策略
# 思路：
# - 从 pathfinder 给出的整条路径（path_points3d）上
# - 按 resolution（步长）均匀采样出一串 interim targets（航点）
# - 每次 step 判断当前位置是否到达当前航点（队列第一个）
# - 到达就 pop 掉，继续下一个
# ----------------------------------------------------------------------
class Even(InterimTarget):
    # ------------------------------------------------------------------
    # 构造函数：需要 pathfinder（提供路径）、intervention（提供当前位置）、resolution（采样间距）、threshold（到达阈值）
    # ------------------------------------------------------------------
    def __init__(
        self,
        pathfinder: Pathfinder,     # 路径规划器：需要先跑过 step()，以更新 path_points3d
        intervention: Intervention,     # 介入环境：用于读 tracking3d[0] 作为当前位置
        resolution: float,          # 中间目标采样分辨率（沿路径每隔多少距离放一个航点）
        threshold: float,           # 判定“到达航点”的距离阈值
    ) -> None:
        self.intervention = intervention
        self.threshold = threshold
        self.pathfinder = pathfinder
        self.resolution = resolution
        self.reached = False
        # self.all_coordinates3d = []

    # ------------------------------------------------------------------
    # step：判断是否到达当前航点（coordinates3d = all_coordinates3d[0]）
    # 到达则 pop 掉第一个航点，并置 reached=True
    # ------------------------------------------------------------------
    def step(self) -> None:
        self.reached = False
        position = self.intervention.fluoroscopy.tracking3d[0]
        if self.coordinates3d is not None:
            position_to_target = self.coordinates3d - position
            dist = np.linalg.norm(position_to_target)
            if dist < self.threshold:
                self.reached = True
                self.all_coordinates3d.pop(0)

    # ------------------------------------------------------------------
    # reset：重新计算整条路径上的等距航点序列
    # ------------------------------------------------------------------
    def reset(self, episode_nr: int = 0, seed: Optional[int] = None) -> None:
        self.all_coordinates3d = self._calc_interim_targets()

    # ------------------------------------------------------------------
    # 计算等距航点：沿 path_points3d 以 resolution 为间隔放点
    # 返回：航点列表（numpy array / list of np.ndarray）
    # ------------------------------------------------------------------
    def _calc_interim_targets(self) -> np.ndarray:
        path_points = self.pathfinder.path_points3d
        path_points = path_points[::-1]
        interim_targets = []
        acc_dist = 0.0
        for point, next_point in zip(path_points[:-1], path_points[1:]):
            length = np.linalg.norm(next_point - point)
            acc_dist += length
            while acc_dist >= self.resolution:
                unit_vector = (next_point - point) / length
                interim_target = next_point - unit_vector * (acc_dist - self.resolution)
                interim_targets.append(interim_target)
                acc_dist -= self.resolution

        interim_targets = interim_targets[::-1]
        return interim_targets
