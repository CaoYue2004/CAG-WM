from typing import List, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
from ....util import EveObject


# =========================
# Branch：一条血管分支（中心线点序列）
# frozen=True：不可变对象（创建后字段不能被普通方式修改）
# eq=True：启用相等比较（用于 set、dict key 等）
# =========================
@dataclass(frozen=True, eq=True)
class Branch(EveObject):
    # 分支名称（如 LAD/LCX/RCA 或 branch_0）
    name: str
    # 分支中心线坐标：shape (N,3)
    # compare=False：比较相等时不直接比较这个 ndarray（因为 ndarray 比较麻烦/不可哈希）
    # repr=True：打印对象时显示该字段（但 __repr__ 被覆盖了，这里基本无影响）
    coordinates: np.ndarray = field(init=True, compare=False, repr=True)
    # _coordinates：把 coordinates 转成“可哈希的 tuple(tuple(...))”
    # init=False：不在构造函数参数里出现
    # compare=True：用于 eq 比较（因为它是 tuple，可比较）
    # repr=False：打印时不显示
    _coordinates: List[Tuple[float, float, float]] = field(
        init=False, default=None, compare=True, repr=False
    )

    # dataclass 创建完成后自动调用：用来规范化/冻结数据
    def __post_init__(self):
        if not isinstance(self.coordinates, np.ndarray):
            coordinates = np.array(self.coordinates)
            object.__setattr__(self, "coordinates", coordinates)
        _coordinates = tuple([tuple(coordinate) for coordinate in self.coordinates])
        self.coordinates.flags.writeable = False
        object.__setattr__(self, "_coordinates", _coordinates)

    def __repr__(self) -> str:
        return self.name

    # 分支坐标的最小值（bounding box 的 low）
    @property
    def low(self) -> np.ndarray:
        return np.min(self.coordinates, axis=0)

    # 分支坐标的最大值（bounding box 的 high）
    @property
    def high(self) -> np.ndarray:
        return np.max(self.coordinates, axis=0)

    # 分支中心线总长度：累加相邻点之间的欧氏距离
    @property
    def length(self) -> float:
        return np.sum(
            np.linalg.norm(self.coordinates[:-1] - self.coordinates[1:], axis=1)
        )

    # 判断点是否“落在分支附近”：如果点到分支任一点距离 < radius，则认为在分支内
    # points: (3,) 或 (M,3)
    # 返回：shape (M,) 的 bool
    def in_branch(self, points: np.ndarray, radius: float) -> np.ndarray:
        # 单点扩维
        if points.ndim == 1:
            points = np.expand_dims(points, 0)
        # 构造广播形状： [N] + [M,3] => (N,M,3)
        broadcast_shape = [self.coordinates.shape[0]] + list(points.shape)
        # 把 points 广播到 (N,M,3)
        points = np.broadcast_to(points, broadcast_shape)
        # 交换轴：变为 (M,N,3)，方便按“每个点”对所有分支点计算距离
        points = np.swapaxes(points, 0, 1)
        # vectors: (M,N,3)，表示每个点到每个分支点的向量
        vectors = points - self.coordinates
        # dist: (M,N)，每个点到每个分支点的距离
        dist = np.linalg.norm(vectors, axis=-1)
        # in_branch: (M,)——只要存在某个分支点距离 < radius 即 True
        in_branch = np.any(dist < radius, axis=-1)
        return in_branch

    # 沿着该分支中心线，从 start 到 end 取一段路径（返回点序列）
    def get_path_along_branch(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        # 计算 start 到分支各点距离
        start_to_branch_dist = np.linalg.norm(self.coordinates - start, axis=1)
        # start 最近的中心线点索引
        start_idx = np.argmin(start_to_branch_dist)

        # 计算 end 到分支各点距离
        end_to_branch_dist = np.linalg.norm(self.coordinates - end, axis=1)
        # end 最近的中心线点索引
        end_idx = np.argmin(end_to_branch_dist)
        # 两个索引差：决定方向（正向/反向）
        idx_diff = end_idx - start_idx
        # 如果 start_idx == end_idx（都落在同一个中心线点附近）
        if abs(idx_diff) == 0:
            # 默认向“正方向”走
            idx_dir = 1
            # 用相邻点避免 start/end 都贴在同一点导致路径退化
            start_idx += 1
            end_idx -= 1
        else:
            # use next idx to prevent hopping between points
            # idx_dir = +1 或 -1，表示沿中心线前进方向
            # 目的：防止“跳点”——用下一点方向更稳定
            idx_dir = int(idx_diff / abs(idx_diff))
            start_idx += idx_dir
            end_idx -= idx_dir

        # 从 start_idx 到 end_idx，步长 idx_dir 取出中心线片段
        partial_branch = self.coordinates[start_idx : end_idx + idx_dir : idx_dir]
        path = np.concatenate(
            [start.reshape(1, 3), partial_branch, end.reshape(1, 3)], axis=0
        )
        return path


# =========================
# BranchWithRadii：每个中心线点还带局部半径
# 继承 Branch，但增加 radii（shape (N,)）
# =========================
@dataclass(frozen=True, eq=True)
class BranchWithRadii(Branch):
    name: str
    coordinates: np.ndarray = field(init=True, compare=False, repr=True)
    radii: np.ndarray = field(init=True, compare=False, repr=True)
    _coordinates: List[Tuple[float, float, float]] = field(
        init=False, default=None, compare=True, repr=False
    )
    _radii: List[Tuple[float, float, float]] = field(
        init=False, default=None, compare=True, repr=False
    )

    def __post_init__(self):
        if not isinstance(self.coordinates, np.ndarray):
            coordinates = np.array(self.coordinates)
            object.__setattr__(self, "coordinates", coordinates)
        if not isinstance(self.radii, np.ndarray):
            radii = np.array(self.radii)
            object.__setattr__(self, "radii", radii)
        _coordinates = tuple([tuple(coordinate) for coordinate in self.coordinates])
        _radii = tuple(self.radii.tolist())
        self.coordinates.flags.writeable = False
        self.radii.flags.writeable = False
        object.__setattr__(self, "_coordinates", _coordinates)
        object.__setattr__(self, "_radii", _radii)

    @property
    def low(self) -> np.ndarray:
        shape = self.coordinates.shape
        radii = np.broadcast_to(self.radii.reshape((-1, 1)), shape)
        coords_low = self.coordinates - radii
        return np.min(coords_low, axis=0)

    @property
    def high(self) -> np.ndarray:
        shape = self.coordinates.shape
        radii = np.broadcast_to(self.radii.reshape((-1, 1)), shape)
        coords_high = self.coordinates + radii
        return np.max(coords_high, axis=0)

    def in_branch(self, points: np.ndarray, radius=None) -> np.ndarray:
        if points.ndim == 1:
            points = np.expand_dims(points, 0)
        broadcast_shape = [self.coordinates.shape[0]] + list(points.shape)
        points = np.broadcast_to(points, broadcast_shape)
        points = np.swapaxes(points, 0, 1)
        vectors = points - self.coordinates
        dist = np.linalg.norm(vectors, axis=-1)
        in_branch = np.any(dist < self.radii, axis=-1)
        return in_branch


# =========================
# BranchingPoint：分叉点（连接若干分支）
# =========================
@dataclass(frozen=True, eq=True)
class BranchingPoint:
    coordinates: np.ndarray = field(init=True, compare=False, repr=True)    # 分叉点坐标
    radius: float       # 分叉“影响半径”
    connections: List[BranchWithRadii]      # 连接的分支列表
    # _coordinates：可哈希坐标缓存，用于 eq 比较
    _coordinates: List[Tuple[float, float, float]] = field(
        init=False, default=None, compare=True, repr=False
    )

    def __post_init__(self):
        # 冻结坐标数组
        self.coordinates.flags.writeable = False
        # 坐标转 tuple 以便可比较/可哈希
        coordinates = tuple(self.coordinates)
        object.__setattr__(self, "_coordinates", coordinates)

    def __repr__(self) -> str:
        return f"BranchingPoint({self.connections})"


# =========================
# calc_branching：给定 branches + 统一/每支半径，计算分叉点列表
# radii 可以是标量（所有分支同半径）或 list（每个分支一个）
# =========================
def calc_branching(branches: List[Branch], radii: Union[float, List[float]]):
    raw_branching_points: List[BranchingPoint] = []
    # 如果 radii 是单个数，则复制成每个分支一个
    if isinstance(radii, (float, int)):
        radii = [radii for branch in branches]

    # 遍历每条 main_branch，并与其它分支比对是否“相交/接近”
    for main_branch, main_radius in zip(branches, radii):
        # find connecting branches
        for other_branch, other_radius in zip(branches, radii):
            # 跳过自己
            if other_branch == main_branch:
                continue

            # 判断 other_branch 的各点是否落在 main_branch 半径范围内
            points_in_main_branch = main_branch.in_branch(
                other_branch.coordinates, main_radius
            )
            # 只要有点落入，就认为存在连接
            if np.any(points_in_main_branch):
                # 找出所有落入的点索引
                idxs = np.argwhere(points_in_main_branch)
                for idx in idxs:
                    # 取连接点坐标（other_branch 的某个点）
                    coords = other_branch.coordinates[idx[0]]
                    # 添加一个原始分叉点（可能有重复/多个点）
                    raw_branching_points.append(
                        BranchingPoint(
                            coords,
                            other_radius,
                            [main_branch, other_branch],
                        )
                    )

    # 合并/去重/聚类这些 raw branching points
    branching_points = _consolidate_branching_points(raw_branching_points)

    return branching_points


# calc_branching_with_radii：分支自带 radii 的版本（每点半径不同）
def calc_branching_with_radii(branches: List[BranchWithRadii]):
    raw_branching_points: List[BranchingPoint] = []

    for main_branch in branches:
        # find connecting branches
        for other_branch in branches:
            if other_branch == main_branch:
                continue

            points_in_main_branch = main_branch.in_branch(other_branch.coordinates)
            if np.any(points_in_main_branch):
                idxs = np.argwhere(points_in_main_branch)
                for idx in idxs:
                    coords = other_branch.coordinates[idx[0]]
                    radius = other_branch.radii[idx[0]]
                    raw_branching_points.append(
                        BranchingPoint(
                            coords,
                            radius,
                            [main_branch, other_branch],
                        )
                    )

    branching_points = _consolidate_branching_points(raw_branching_points)

    return branching_points


# =========================
# _consolidate_branching_points：把原始分叉点合并成更干净的列表
# 两阶段：
# 1) 按“连接集合相同”合并并平均坐标
# 2) 按“空间距离重叠（半径和）”再次合并，合并连接集合
# =========================
def _consolidate_branching_points(raw_branching_points: List[BranchingPoint]):
    branching_points: List[BranchingPoint] = []
    # remove duplicates from branching_point_list
    while raw_branching_points:
        branching_point = raw_branching_points.pop(-1)
        to_average = [branching_point]
        for i, other_branching_point in enumerate(raw_branching_points):
            if set(other_branching_point.connections) == set(
                branching_point.connections
            ):
                # add connections of branching point to other branching point
                to_average.append(other_branching_point)

        for bp in to_average[1:]:
            raw_branching_points.remove(bp)
        coords = np.array([0.0, 0.0, 0.0])
        main_radius = np.inf
        for bp in to_average:
            coords += bp.coordinates
            main_radius = min(bp.radius, main_radius)
        coords /= len(to_average)
        branching_points.append(
            BranchingPoint(coords, main_radius, tuple(branching_point.connections))
        )

    raw_branching_points = branching_points
    branching_points: List[BranchingPoint] = []

    while raw_branching_points:
        branching_point = raw_branching_points.pop(-1)
        discard_branching_point = False
        for i, other_branching_point in enumerate(raw_branching_points):
            distance = np.linalg.norm(
                branching_point.coordinates - other_branching_point.coordinates
            )
            check_distance = (
                distance < branching_point.radius + other_branching_point.radius
            )

            if check_distance:
                # add connections of branching point to other branching point
                coord = branching_point.coordinates + other_branching_point.coordinates
                coord /= 2
                main_radius = max(branching_point.radius, other_branching_point.radius)
                connections = list(branching_point.connections) + list(
                    other_branching_point.connections
                )
                new_branching_point = BranchingPoint(
                    coord,
                    main_radius,
                    tuple(set(connections)),
                )
                raw_branching_points[i] = new_branching_point

                discard_branching_point = True

        if not discard_branching_point:
            branching_points.append(branching_point)
    return branching_points


# =========================
# 缩放：xyz 三个方向缩放分支坐标（半径不变）
# =========================
def scale_branches_xyz(
    branches: List[Branch], xyz_scaling: Tuple[float, float, float]
) -> Tuple[Branch]:
    xyz_scaling = np.array(xyz_scaling, dtype=np.float32)
    new_branches = []
    for branch in branches:
        new_coordinates = branch.coordinates * xyz_scaling
        if isinstance(branch, BranchWithRadii):
            new_branch = BranchWithRadii(branch.name, new_coordinates, branch.radii)
        else:
            new_branch = Branch(branch.name, new_coordinates)
        new_branches.append(new_branch)
    return tuple(new_branches)


def scale_branches_d(
    branches: List[BranchWithRadii], d_scaling: float
) -> Tuple[BranchWithRadii]:
    new_branches = []
    for branch in branches:
        new_radii = branch.radii * d_scaling
        new_branches.append(BranchWithRadii(branch.name, branch.coordinates, new_radii))
    return tuple(new_branches)


def scale_branches_xyzd(
    branches: List[BranchWithRadii], xyzd_scaling: Tuple[float, float, float, float]
) -> Tuple[BranchWithRadii]:
    branches = scale_branches_xyz(branches, xyzd_scaling[0:3])
    return scale_branches_d(branches, xyzd_scaling[-1])


# =========================
# 旋转：对每条分支坐标做旋转（可用于模拟 LAO/RAO、CRA/CAU 等）
# rotate_yzx_deg：(y,z,x) 三个角度（度）
# =========================
def rotate_branches(
    branches: List[Branch], rotate_yzx_deg: Tuple[float, float, float]
) -> Tuple[Branch]:
    new_branches = []
    for branch in branches:
        new_coordinates = rotate_array(
            array=branch.coordinates,
            y_deg=rotate_yzx_deg[0],
            z_deg=rotate_yzx_deg[1],
            x_deg=rotate_yzx_deg[2],
        )
        if isinstance(branch, BranchWithRadii):
            new_branch = BranchWithRadii(branch.name, new_coordinates, branch.radii)
        else:
            new_branch = Branch(branch.name, new_coordinates)
        new_branches.append(new_branch)
    return tuple(new_branches)


def omit_branches_axis(
    branches: List[Branch], axis_to_remove: str, dummy_value: float = 0
) -> Tuple[Branch]:
    if axis_to_remove not in ["x", "y", "z"]:
        raise ValueError(f"to_2d() {axis_to_remove =} has to be 'x', 'y' or 'z'")
    convert = {"x": 0, "y": 1, "z": 2}
    axis_to_remove = convert[axis_to_remove]
    new_branches = []
    for branch in branches:
        new_coordinates = np.delete(branch.coordinates, axis_to_remove, axis=1)
        new_coordinates = np.insert(
            new_coordinates, axis_to_remove, dummy_value, axis=1
        )
        if isinstance(branch, BranchWithRadii):
            new_branch = BranchWithRadii(branch.name, new_coordinates, branch.radii)
        else:
            new_branch = Branch(branch.name, new_coordinates)
        new_branches.append(new_branch)
    return tuple(new_branches)


def rotate_array(
    array: np.ndarray,
    y_deg: float,
    z_deg: float,
    x_deg: float,
):
    y_rad = y_deg * np.pi / 180
    lao_rao_rad = z_deg * np.pi / 180
    cra_cau_rad = x_deg * np.pi / 180

    rotation_matrix_y = np.array(
        [
            [np.cos(y_rad), 0, np.sin(y_rad)],
            [0, 1, 0],
            [-np.sin(y_rad), 0, np.cos(y_rad)],
        ],
        dtype=np.float32,
    )

    rotation_matrix_lao_rao = np.array(
        [
            [np.cos(lao_rao_rad), -np.sin(lao_rao_rad), 0],
            [np.sin(lao_rao_rad), np.cos(lao_rao_rad), 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    rotation_matrix_cra_cau = np.array(
        [
            [1, 0, 0],
            [0, np.cos(cra_cau_rad), -np.sin(cra_cau_rad)],
            [0, np.sin(cra_cau_rad), np.cos(cra_cau_rad)],
        ],
        dtype=np.float32,
    )
    rotation_matrix = np.matmul(rotation_matrix_cra_cau, rotation_matrix_lao_rao)
    rotation_matrix = np.matmul(rotation_matrix, rotation_matrix_y)
    # transpose such that matrix multiplication works
    rotated_array = np.matmul(rotation_matrix, array.T).T
    return rotated_array
