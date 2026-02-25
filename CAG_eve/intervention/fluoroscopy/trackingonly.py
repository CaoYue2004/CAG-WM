import logging
import gymnasium as gym
import numpy as np
from typing import List, Optional, Tuple
from .fluoroscopy import SimulatedFluoroscopy
from ...util.coordtransform import (
    vessel_cs_to_tracking3d,
    tracking3d_to_2d,
)
from ..simulation import Simulation
from ..vesseltree import VesselTree


# 定义 TrackingOnly：一种“仅输出 tracking（位置），不输出真实图像”的透视模块
class TrackingOnly(SimulatedFluoroscopy):
    # 初始化：绑定仿真器、血管树，并设置成像参数（频率、旋转、中心、视野）
    def __init__(
        self,
        simulation: Simulation,     # simulation：物理仿真对象（SOFA 等），提供器械 DOF 位置、插入长度等
        vessel_tree: VesselTree,    # vessel_tree：血管树对象（提供坐标空间、中心线等）
        image_frequency: float = 7.5,       # image_frequency：成像频率（Hz），例如 7.5Hz
        image_rot_zx: Optional[Tuple[float, float]] = None,     # image_rot_zx：成像坐标旋转参数（zx 两个角），可为 None
        image_center: Optional[Tuple[float, float, float]] = None,      # image_center：成像中心（3D），可为 None
        field_of_view: Optional[Tuple[float, float]] = None,        # field_of_view：视野范围（可能影响坐标映射/裁剪），可为 None
    ) -> None:
        self.logger = logging.getLogger(self.__module__)
        self.simulation = simulation
        self.vessel_tree = vessel_tree
        self.image_rot_zx = image_rot_zx or [0,0]
        self.image_frequency = image_frequency
        self.image_center = image_center or [0,0,0]
        self.field_of_view = field_of_view

    # image_space：定义图像 observation space（这里是一个 1x1 的 dummy 图像）
    @property
    def image_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(1, 1, (1, 1), dtype=np.uint8)     # 返回一个 Box 空间：形状 (1,1)，dtype uint8（占位用）

    # image：返回当前图像（这里永远是 [[1]]，即 dummy 图像）
    @property
    def image(self) -> np.ndarray:
        return np.array([[1]], dtype=np.uint8)      # 返回一个固定的 1x1 uint8 图像

    # tracking3d_space：tracking3d 的全局空间范围（基于 vessel_tree 的 coordinate_space 映射后得到）
    @property
    def tracking3d_space(self) -> gym.spaces.Box:
        # 取血管树坐标空间的下界
        low = self.vessel_tree.coordinate_space.low
        # 取血管树坐标空间的上界
        high = self.vessel_tree.coordinate_space.high
        # 将 low 从 vessel 坐标系映射到 tracking3d 坐标系（考虑旋转/中心/视野）
        low = vessel_cs_to_tracking3d(
            low, self.image_rot_zx, self.image_center, self.field_of_view
        )
        # 将 high 从 vessel 坐标系映射到 tracking3d 坐标系
        high = vessel_cs_to_tracking3d(
            high, self.image_rot_zx, self.image_center, self.field_of_view
        )
        # 返回 tracking3d 的 Box 空间
        return gym.spaces.Box(low=low, high=high)

    # tracking3d_space_episode：每个 episode 更紧的 tracking3d 空间（只基于中心线坐标，并加一点 margin）
    @property
    def tracking3d_space_episode(self) -> gym.spaces.Box:
        coords = self.vessel_tree.centerline_coordinates
        coords = vessel_cs_to_tracking3d(
            coords, self.image_rot_zx, self.image_center, self.field_of_view
        )
        low = np.min(coords, axis=0)
        low -= 0.1 * np.abs(low)
        high = np.max(coords, axis=0)
        high += 0.1 * np.abs(high)
        return gym.spaces.Box(low=low, high=high)

    # tracking2d_space：tracking2d 的全局空间（把 tracking3d_space 的 low/high 投影到 2D）
    @property
    def tracking2d_space(self) -> gym.spaces.Box:
        space_3d = self.tracking3d_space
        low = tracking3d_to_2d(space_3d.low)
        high = tracking3d_to_2d(space_3d.high)
        return gym.spaces.Box(low=low, high=high)

    # tracking2d_space_episode：episode 内更紧的 2D tracking 空间（基于 tracking3d_space_episode）
    @property
    def tracking2d_space_episode(self) -> gym.spaces.Box:
        space_3d = self.tracking3d_space_episode
        low = tracking3d_to_2d(space_3d.low)
        high = tracking3d_to_2d(space_3d.high)
        return gym.spaces.Box(low=low, high=high)

    # tracking3d：返回当前时刻所有 DOF 的 3D tracking 坐标（由仿真 DOF 位置映射而来）
    @property
    def tracking3d(self) -> np.ndarray:
        # 把仿真中的 dof_positions（一般是世界/血管坐标系）转换到 tracking3d 坐标系
        return vessel_cs_to_tracking3d(
            self.simulation.dof_positions,
            self.image_rot_zx,
            self.image_center,
            self.field_of_view,
        )

    # tracking2d：返回当前 tracking3d 投影到 2D 的结果
    @property
    def tracking2d(self) -> np.ndarray:
        return tracking3d_to_2d(self.tracking3d)

    # device_trackings3d：返回每个器械各自的 3D tracking（按插入长度切出每个器械的前段轨迹）
    @property
    def device_trackings3d(self) -> List[np.ndarray]:
        position = self.tracking3d      # 取出当前所有 DOF 的 3D 位置
        position = np.flip(position)        # 将点序列翻转（通常是为了让序列从“插入端→尖端”或相反方向一致）
        point_diff = position[:-1] - position[1:]       # 计算相邻点差向量（N-1,3）
        length_btw_points = np.linalg.norm(point_diff, axis=-1)     # 计算相邻点之间的欧式距离（N-1）
        cum_length = np.cumsum(length_btw_points)       # 计算从起点开始的累积弧长（N-1）
        inserted_lengths = self.simulation.inserted_lengths     # 获取每个器械当前插入长度（列表长度 = 器械数）

        d_lengths = np.array(inserted_lengths)      # 将插入长度转为 numpy 数组
        n_devices = d_lengths.size      # 器械数量
        n_dofs = cum_length.size        # DOF 段数量（等于 cum_length 的长度）
        # 将 d_lengths 广播成 (n_devices, n_dofs)，用于与 cum_length 对齐做差
        d_lengths = np.broadcast_to(d_lengths, (n_dofs, n_devices)).transpose()
        cum_length = np.broadcast_to(cum_length, (n_devices, n_dofs))

        # 计算每个器械的插入长度与每个弧长位置的差值（越小越接近该插入位置）
        diff = np.abs(cum_length - d_lengths)
        # 对每个器械，找到差值最小的索引 idx（即最接近插入长度的位置）
        idxs = np.argmin(diff, axis=1)

        # 对每个器械：取 position[:idx+1] 作为该器械目前“有效长度”的点序列，并翻转回原方向
        trackings = [np.flip(position[: idx + 1]) for idx in idxs]
        return trackings

    # device_trackings2d：返回每个器械对应的 2D tracking（对 device_trackings3d 逐个投影）
    @property
    def device_trackings2d(self) -> List[np.ndarray]:
        trackings_3d = self.device_trackings3d
        trackings_2d = [tracking3d_to_2d(tracking) for tracking in trackings_3d]
        return trackings_2d
