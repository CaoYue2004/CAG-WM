from abc import ABC
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
from ...util import EveObject
from .sofadevice import SOFADevice, NonProceduralShape
from ..vesseltree.util.meshing import get_temp_mesh_path


# -------------------------
# 抽象设备基类：定义“设备”应当具备的字段/接口
# -------------------------
class Device(EveObject, ABC):
    name: str       # 设备名字（例如 guidewire/catheter）
    sofa_device: SOFADevice     # 对应的 SOFA 侧设备对象（具体实现由子类给出）
    velocity_limit: Tuple[float, float]     # 速度限制(例如前进/后退最大速度)
    length: float       # 总长度
    diameter: float     # 直径


# 角度（度）-> 弧度（SOFA/几何计算一般用弧度）
def deg_to_rad(deg: float) -> float:
    return deg * np.pi / 180


# -------------------------
# 直线段定义：长度 + 采样/离散化密度参数
# -------------------------
@dataclass
class StraightPart:
    length: float       # 直线段长度（单位看你系统，一般是 mm）
    visu_edges_per_mm: float    # 可视化网格：每 mm 多少条边（离散越密越细）
    collis_edges_per_mm: float  # 碰撞网格：每 mm 多少条边
    beams_per_mm: float         # Beam 密度：每 mm 多少 beam 单元


# -------------------------
# 圆弧段定义：半径 + 平面内/平面外转角 + 离散化密度
# -------------------------
@dataclass
class Arc:
    radius: float       # 圆弧半径
    angle_in_plane_deg: float       # “平面内”转角（单位：度）
    angle_out_of_plane_deg: float   # “出平面”转角（单位：度）
    visu_edges_per_mm: float        # 可视化边密度
    collis_edges_per_mm: float      # 碰撞边密度
    beams_per_mm: float             # beam 密度
    resolution: float = 0.1         # 采样步长（沿弧长的采样分辨率，越小点越密）


# -------------------------
# 把点云存成 OBJ 的折线（v + l）
# 仅用作“中心线/路径”的线段网格，而不是管状表面
# -------------------------
def save_line_mesh(point_cloud: np.ndarray, file: str):
    # 打开文件写入（OBJ 文本格式）
    with open(file, "w", encoding="utf-8") as f:
        vertices = [
            f"v {point[0]:.4f} {point[1]:.4f} {point[2]:.4f}\n" for point in point_cloud
        ]
        f.writelines(vertices)
        connections = [f"l {i+1} {i+2}\n" for i in range(point_cloud.shape[0] - 1)]
        f.writelines(connections)


# -------------------------
# MeshDevice：用 StraightPart/Arc 拼出设备中心线 -> 生成折线 mesh -> 交给 SOFA 设备构建
# -------------------------
class MeshDevice(Device):
    def __init__(
        self,
        elements: List[Union[StraightPart, Arc]],   # 设备由若干段直线/圆弧组成
        outer_diameter: float,      # 外径
        inner_diameter: float,      # 内径（空心器械，比如导管）
        poisson_ratio: float,       # 泊松比
        young_modulus: float,       # 杨氏模量
        mass_density: float,        # 密度
        color: Tuple[float, float, float],      # 颜色（RGB）
    ):
        # 保存构型与材料参数
        self.elements = elements
        self.outer_diameter = outer_diameter
        self.inner_diameter = inner_diameter
        self.poisson_ratio = poisson_ratio
        self.young_modulus = young_modulus
        self.mass_density = mass_density
        self.color = color

        # 生成中心线点云 + 每段关键点位置 + 每段离散化参数
        (
            point_cloud,
            key_points,
            visu_edges,
            collis_edges,
            beams,
        ) = self._create_shape_point_cloud()
        # 把中心线点云写成临时 mesh 文件（OBJ 折线）
        mesh_path = self._create_mesh(point_cloud)

        # 由外径/内径得到半径（SOFA 里常用 radius）
        radius = self.outer_diameter / 2
        inner_radius = self.inner_diameter / 2
        # 总长度：key_points 最后一个就是累计长度
        length = key_points[-1]

        # 构建 SOFA 侧的 NonProceduralShape（非程序化形状：从 mesh 文件读）
        # 把 key_points、离散参数等都传进去，供 SOFA 内部生成 beam/碰撞/可视化结构
        self.sofa_device = NonProceduralShape(
            mesh_path=mesh_path,
            length=length,
            poisson_ratio=self.poisson_ratio,
            young_modulus=self.young_modulus,
            radius=radius,
            inner_radius=inner_radius,
            mass_density=self.mass_density,
            num_edges=visu_edges,
            num_edges_collis=collis_edges,
            density_of_beams=beams,
            key_points=key_points,
            color=self.color,
        )

    # -------------------------
    # 核心：根据 elements 拼出中心线点云
    # 同时计算 key_points（每段结束时的累计长度）和每段离散参数
    # -------------------------
    def _create_shape_point_cloud(self) -> np.ndarray:
        # 初始“平面内轴”和“出平面轴”
        # 这里用两个互相正交的轴来定义弯曲方向的组合
        in_plane_axis = np.array([0, 0, 1])         # 默认平面内轴：Z
        out_of_plane_axis = np.array([0, 1, 0])     # 默认出平面轴：Y

        # 初始点：原点
        last_point = np.array([0.0, 0.0, 0.0])
        # 初始方向：沿 X 正方向前进
        direction = np.array([1.0, 0.0, 0.0])
        # key_points 记录：从起点到每一段终点的累计长度
        key_points = [0.0]
        # 每段的可视化边数/碰撞边数/beam 数（通常与长度相关）
        visu_edges = []
        collis_edges = []
        beams = []
        # point_clouds 是分段点云列表，最后 concat
        # 先放起点（shape: (1,3)）
        point_clouds = [last_point.reshape(1, -1)]

        # 逐段拼装
        for element in self.elements:
            # --------- 弧段 ---------
            if isinstance(element, Arc):
                (
                    last_point,         # 更新后的末端点
                    direction,          # 更新后的前进方向
                    in_plane_axis,      # 更新后的平面内轴（会随弯曲旋转）
                    out_of_plane_axis,  # 更新后的出平面轴（会随弯曲旋转）
                    point_clouds,       # 增加了该段采样点
                ) = self._add_curve_part(
                    element,
                    last_point,
                    direction,
                    in_plane_axis,
                    out_of_plane_axis,
                    point_clouds,
                )

                # 通过点云差分估算该段弧长（逐段距离求和）
                pc_diff = point_clouds[-1][:-1] - point_clouds[-1][1:]
                lengths = np.linalg.norm(pc_diff, axis=-1)
                length = np.sum(lengths)
            # --------- 直段 ---------
            elif isinstance(element, StraightPart):
                last_point, direction, point_clouds = self._add_straight_part(
                    element, last_point, direction, point_clouds
                )
                # 直线段长度直接取定义值
                length = element.length
            # 更新累计长度 key_points
            key_points.append(key_points[-1] + length)

            # 根据“每 mm 边密度/beam 密度”换算每段整数数量
            # ceil：保证至少够密，不会因为小数截断变少
            visu_edges.append(int(np.ceil(length * element.visu_edges_per_mm)))
            collis_edges.append(int(np.ceil(length * element.collis_edges_per_mm)))
            beams.append(int(np.ceil(length * element.beams_per_mm)))

        # 把所有分段点云拼起来
        point_cloud = np.concatenate(point_clouds, axis=0)

        return (
            point_cloud,
            tuple(key_points),
            tuple(visu_edges),
            tuple(collis_edges),
            tuple(beams),
        )

    # -------------------------
    # 增加一段直线
    # -------------------------
    def _add_straight_part(
        self,
        straight_element: StraightPart,
        last_point: np.ndarray,
        direction: np.ndarray,
        point_clouds: List[np.ndarray],
    ) -> None:
        # 取直线段长度
        length = straight_element.length
        # 起点就是上一段末端点
        start = last_point

        # 沿着 [0, length] 取采样点
        # 这里写死为 2 个点：0 和 length（意味着每个直线段只加一个新点）
        sample_points: np.ndarray = np.linspace(0.0, length, 2, endpoint=True)
        # 去掉第一个点（0），避免重复 start 点
        sample_points = sample_points[1:]
        # 生成点云数组 (N,3)
        shape = (sample_points.shape[0], 3)
        # 每个采样点先设为 direction
        point_cloud = np.full(shape, direction)
        # 缩放：direction * 距离
        point_cloud *= sample_points[:, None]
        # 平移：加起点
        point_cloud += start
        # 数值保留 4 位小数（避免浮点小噪声）
        point_cloud = np.round(point_cloud, 4)

        # 更新 last_point 为该段末端
        last_point = point_cloud[-1]
        # 追加该段点云
        point_clouds.append(point_cloud)
        return last_point, direction, point_clouds

    # -------------------------
    # 增加一段圆弧（允许同时有“平面内”和“出平面”的角度）
    # -------------------------
    def _add_curve_part(
        self,
        arc_def: Arc,
        last_point: np.ndarray,
        direction: np.ndarray,
        in_plane_axis: np.ndarray,
        out_of_plane_axis: np.ndarray,
        point_clouds: List[np.ndarray],
    ) -> None:
        start = last_point
        initial_direction = direction
        angle_in_plane = deg_to_rad(arc_def.angle_in_plane_deg)
        angle_out_of_plane = deg_to_rad(arc_def.angle_out_of_plane_deg)
        radius = arc_def.radius
        resolution = arc_def.resolution

        angle, axis = self._get_combined_angle_axis(
            angle_in_plane, angle_out_of_plane, in_plane_axis, out_of_plane_axis
        )

        dir_to_curve_center = self._rotate_around_axis(
            initial_direction, np.pi / 2, axis
        )
        curve_center = start + dir_to_curve_center * radius

        arc_length = radius * abs(angle)
        n_points = int(np.ceil(arc_length / resolution)) + 1
        sample_angles = np.linspace(0.0, angle, n_points, endpoint=True)
        sample_angles = sample_angles[1:]

        base_vector = -dir_to_curve_center * radius
        vectors = [
            self._rotate_around_axis(base_vector, angle, axis)
            for angle in sample_angles
        ]
        vectors = np.array(vectors)

        curve_point_cloud = vectors + curve_center
        curve_point_cloud = np.round(curve_point_cloud, 4)
        direction = self._rotate_around_axis(initial_direction, angle, axis)
        out_of_plane_axis = self._rotate_around_axis(out_of_plane_axis, angle, axis)
        in_plane_axis = self._rotate_around_axis(in_plane_axis, angle, axis)
        last_point = curve_point_cloud[-1]
        point_clouds.append(curve_point_cloud)

        return last_point, direction, in_plane_axis, out_of_plane_axis, point_clouds

    # -------------------------
    # 将“平面内角 + 出平面角”合成为一个等效的旋转轴和旋转角
    # -------------------------
    def _get_combined_angle_axis(
        self,
        in_plane_angle: float,
        out_of_plane_angle: float,
        in_plane_axis: np.ndarray,
        out_of_plane_axis: np.ndarray,
    ):
        axis = (
            in_plane_axis * in_plane_angle + out_of_plane_axis * out_of_plane_angle
        ) / (abs(in_plane_angle) + abs(out_of_plane_angle))
        angle = (in_plane_angle**2 + out_of_plane_angle**2) / (
            abs(in_plane_angle) + abs(out_of_plane_angle)
        )

        return angle, axis

    # -------------------------
    # Rodrigues 旋转公式：vector 绕 axis 旋转 angle
    # -------------------------
    @staticmethod
    def _rotate_around_axis(vector: np.ndarray, angle: float, axis: np.ndarray):
        axis = axis / np.linalg.norm(axis)
        x, y, z = tuple(axis)
        cos = np.cos(angle)
        sin = np.sin(angle)
        R = np.array(
            [
                [
                    cos + x**2 * (1 - cos),
                    x * y * (1 - cos) - z * sin,
                    x * z * (1 - cos) + y * sin,
                ],
                [
                    y * x * (1 - cos) + z * sin,
                    cos + y**2 * (1 - cos),
                    y * z * (1 - cos) - x * sin,
                ],
                [
                    z * x * (1 - cos) - y * sin,
                    z * y * (1 - cos) + x * sin,
                    cos + z**2 * (1 - cos),
                ],
            ]
        )

        return np.matmul(R, vector)

    # -------------------------
    # 把中心线点云写到临时 mesh 文件里（OBJ折线），返回路径
    # -------------------------
    def _create_mesh(self, device_point_cloud: np.ndarray) -> str:
        mesh_path = get_temp_mesh_path("endovascular_instrument")
        save_line_mesh(device_point_cloud, mesh_path)
        return mesh_path
