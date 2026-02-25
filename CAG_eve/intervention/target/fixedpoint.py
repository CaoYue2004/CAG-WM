"""
固定三维点目标类：用于指定精确的 3D 坐标作为导航目标。

与 CenterlineRandom 的区别：
- CenterlineRandom：从血管中心线随机采样目标点
- FixedPoint3D：直接指定精确的 tracking 坐标系 3D 点

用法示例：
    target = FixedPoint3D(
        fluoroscopy=fluoro,
        threshold=2.0,
        default_point=[0.312, -0.045, 0.187]
    )

    # reset 时可以传入新坐标覆盖 default_point
    target.reset(coordinates=[0.285, 0.012, 0.203])
"""

from typing import Optional
import numpy as np

from .target import Target
from ..fluoroscopy import Fluoroscopy
from ...util.coordtransform import vessel_cs_to_tracking3d, tracking3d_to_2d


class FixedPoint3D(Target):
    """
    固定三维点目标。

    参数:
        fluoroscopy: Fluoroscopy 对象，用于读取当前器械位置和坐标转换
        threshold: 到达判定阈值（单位：mm，距离 < threshold 视为到达）
        default_point: 默认目标点的 tracking 坐标系 3D 坐标 [x, y, z]
                       reset() 时如果不传 coordinates 参数则使用此默认值
    """

    def __init__(
            self,
            fluoroscopy: Fluoroscopy,
            threshold: float,
            default_point: Optional[np.ndarray] = None,
    ) -> None:
        self.fluoroscopy = fluoroscopy
        self.threshold = threshold

        # 默认目标点（如果不传则设为原点）
        if default_point is None:
            self.default_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            self.default_point = np.asarray(default_point, dtype=np.float32).copy()

        # 初始化状态（会在 reset 时正式设置）
        target_vessel_cs = self.default_point.copy()
        self.coordinates3d = vessel_cs_to_tracking3d(
            target_vessel_cs,
            self.fluoroscopy.image_rot_zx,
            self.fluoroscopy.image_center,
            self.fluoroscopy.field_of_view,
        )
        self.reached = False

    @property
    def coordinates2d(self) -> np.ndarray:
        """将 3D tracking 坐标投影到 2D 透视平面。"""
        return tracking3d_to_2d(self.coordinates3d)

    def reset(
            self,
            episode_nr: int = 0,
            seed: Optional[int] = None,
            coordinates: Optional[np.ndarray] = None,
    ) -> None:
        """
        重置目标点。

        参数:
            episode_nr: episode 编号（本类未使用，保留接口兼容性）
            seed: 随机种子（本类未使用，保留接口兼容性）
            coordinates: 指定的目标点 3D 坐标 [x, y, z]。
                        如果为 None，则使用 default_point。
        """
        if coordinates is not None:
            target_vessel_cs = np.asarray(coordinates, dtype=np.float32).copy()
            self.coordinates3d = vessel_cs_to_tracking3d(
                target_vessel_cs,
                self.fluoroscopy.image_rot_zx,
                self.fluoroscopy.image_center,
                self.fluoroscopy.field_of_view,
            )
        else:
            target_vessel_cs = self.default_point.copy()
            self.coordinates3d = vessel_cs_to_tracking3d(
                target_vessel_cs,
                self.fluoroscopy.image_rot_zx,
                self.fluoroscopy.image_center,
                self.fluoroscopy.field_of_view,
            )

        self.reached = False

    def set_target(self, coordinates: np.ndarray) -> None:
        """
        便捷方法：直接设置目标点（等价于 reset(coordinates=...)）。

        参数:
            coordinates: 目标点 3D 坐标 [x, y, z]
        """
        self.reset(coordinates=coordinates)

