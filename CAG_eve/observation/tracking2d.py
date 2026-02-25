import numpy as np

from ..intervention import Intervention
from .observation import Observation, gym


# ----------------------------------------------------------------------
# Tracking2D：2D 追踪点观测
# 作用：
# - 从 fluoroscopy.tracking2d 取得一串 2D 轨迹点（通常是导丝/导管在透视图上的投影轨迹）
# - 沿这条轨迹按固定间距 resolution 采样出 n_points 个点
# - 输出形状为 (n_points, 2) 或 (n_points, D)（取决于 tracking2d 的维度）
# ----------------------------------------------------------------------
class Tracking2D(Observation):
    # ------------------------------------------------------------------
    # 构造函数
    # - n_points：希望输出多少个采样点
    # - resolution：采样间距（沿轨迹累计距离每达到 resolution 取一个点）
    # - name：观测名称
    # ------------------------------------------------------------------
    def __init__(
        self,
        intervention: Intervention,
        n_points: int = 2,
        resolution: float = 1.0,
        name: str = "tracking2d",
    ) -> None:
        self.name = name
        self.intervention = intervention
        self.n_points = n_points
        self.resolution = resolution
        self.obs = None

    # ------------------------------------------------------------------
    # space：观测空间（Box）
    # 说明：
    # - fluoroscopy.tracking2d_space 定义了单个 2D 点的范围 low/high
    # - 这里通过 np.tile 将其扩展为 n_points 个点的范围
    # ------------------------------------------------------------------
    @property
    def space(self) -> gym.spaces.Box:
        low = self.intervention.fluoroscopy.tracking2d_space.low
        high = self.intervention.fluoroscopy.tracking2d_space.high
        low = np.tile(low, [self.n_points, 1])
        high = np.tile(high, [self.n_points, 1])
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    # ------------------------------------------------------------------
    # step：更新观测
    # - 读取 fluoroscopy.tracking2d（整条轨迹）
    # - 调用均匀采样函数输出固定数量的点
    # ------------------------------------------------------------------
    def step(self) -> None:
        tracking = self.intervention.fluoroscopy.tracking2d
        self.obs = self._evenly_distributed_tracking(tracking)

    def reset(self, episode_nr: int = 0) -> None:
        self.step()

    # ------------------------------------------------------------------
    # _evenly_distributed_tracking：
    # 沿 tracking 轨迹从起点开始按距离间隔 resolution 取点，最多取 n_points 个
    # 若轨迹不够长或中途停止，则用最后一个点重复填充到 n_points
    # ------------------------------------------------------------------
    def _evenly_distributed_tracking(self, tracking: np.ndarray) -> np.ndarray:
        tracking = list(tracking)
        tracking_state = [tracking[0]]
        if self.n_points > 1:
            acc_dist = 0.0
            for point, next_point in zip(tracking[:-1], tracking[1:]):
                if len(tracking_state) >= self.n_points or np.all(point == next_point):
                    break
                length = np.linalg.norm(next_point - point)
                dist_to_point = self.resolution - acc_dist
                acc_dist += length
                while (
                    acc_dist >= self.resolution and len(tracking_state) < self.n_points
                ):
                    unit_vector = (next_point - point) / length
                    tracking_point = point + unit_vector * dist_to_point
                    tracking_state.append(tracking_point)
                    acc_dist -= self.resolution

            while len(tracking_state) < self.n_points:
                tracking_state.append(tracking_state[-1])
        return np.array(tracking_state, dtype=np.float32)
