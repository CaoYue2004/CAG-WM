from xml.dom import minidom
from typing import List, Optional, Tuple, Union
import os
import numpy as np
import pyvista as pv

from .vesseltree import VesselTree, Insertion, gym
from .util.branch import Branch, calc_branching, rotate_branches
from .util import calc_insertion
from .util.meshing import get_temp_mesh_path
from .util.vmrdownload import download_vmr_files


SCALING_FACTOR = 1
LOW_HIGH_BUFFER = 3


def _get_branches(model_dir, vtu_mesh, check_if_points_in_mesh: bool) -> List[Branch]:
    # 从模型目录读取所有 .pth 中心线路径文件，转成 Branch 列表
    path_dir = os.path.join(model_dir, "Paths")
    files = _get_available_pths(path_dir)
    files = sorted(files)
    branches = []
    for file in files:
        file_path = os.path.join(path_dir, file)
        branch = _load_points_from_pth(file_path, vtu_mesh, check_if_points_in_mesh)
        if branch.coordinates.size > 3:
            branches.append(branch)
    return branches


def _get_available_pths(directory: str) -> List[str]:
    pth_files = []
    for file in os.listdir(directory):
        if file.endswith(".pth"):
            pth_files.append(file)
    return pth_files


def _get_vtk_file(directory: str, file_ending: str) -> str:
    for file in os.listdir(directory):
        if file.endswith(file_ending):
            path = os.path.join(directory, file)
            return path


def _load_points_from_pth(
    pth_file_path: str, vtu_mesh: pv.UnstructuredGrid, check_if_points_in_mesh: bool
) -> Branch:
    # 读取单个 pth 文件，解析其中的 <pos x= y= z=> 点，生成 Branch
    points = []
    name = os.path.basename(pth_file_path)[:-4]
    # print(f'name={name}')
    with open(pth_file_path, "r", encoding="utf-8") as file:
        next(file)
        next(file)
        tree = minidom.parse(file)

    xml_points = tree.getElementsByTagName("pos")
    # print(f'xml_points={len(xml_points)}')

    points = []
    low = np.array([vtu_mesh.bounds[0], vtu_mesh.bounds[2], vtu_mesh.bounds[4]])
    high = np.array([vtu_mesh.bounds[1], vtu_mesh.bounds[3], vtu_mesh.bounds[5]])
    # print(f'low={low}, high={high}')
    low += LOW_HIGH_BUFFER
    high -= LOW_HIGH_BUFFER
    for point in xml_points:
        # print(f'point={point}')
        x = float(point.attributes["x"].value) * SCALING_FACTOR
        y = float(point.attributes["y"].value) * SCALING_FACTOR
        z = float(point.attributes["z"].value) * SCALING_FACTOR
        # print(f'x={x}, y={y}, z={z}')
        '''if np.any([x, y, z] < low) or np.any([x, y, z] > high):
            continue'''
        points.append([x, y, z])
        # print(f'points={points}')
    points = np.array(points, dtype=np.float32)
    # print(f'points={points}')
    # print(f'check_if_points_in_mesh={check_if_points_in_mesh}')
    if check_if_points_in_mesh:
        to_keep = vtu_mesh.find_containing_cell(points) + 1
        to_keep = np.argwhere(to_keep)
        points = points[to_keep.reshape(-1)]
    # print(f'points={points}')
    return Branch(
        name=name.lower(),
        coordinates=np.array(points, dtype=np.float32),
    )


class CoronaryArtery(VesselTree):
    """
        基于 VMR 数据集构建的血管树：
        - 从 VMR 下载模型
        - 读取中心线分支
        - 计算插入点 / 分叉点
        - 生成可视化 mesh
        """
    def __init__(
        self,
        model_folder: str,     # VMR 模型名称（用于 download_vmr_files）
        insertion_vessel_name: str,     # 插入所在分支名称（例如 femoral 等）
        insertion_point_idx: int,       # 插入点在该分支中心线点序列中的索引
        insertion_direction_idx_diff: int,      # 插入方向用 insertion_point_idx + diff 的点来估计方向
        approx_branch_radii: Union[List[float], float],     # 分支半径的近似值（用于分叉检测阈值等），可以是标量或列表
        rotate_yzx_deg: Optional[Tuple[float, float, float]] = None,        # 可选：对整个模型做旋转（顺序 y,z,x），单位度
        check_if_points_in_mesh: bool = True,       # 是否检查中心线点确实落在 mesh 体内
    ) -> None:
        # 插入相关参数
        self.insertion_point_idx = insertion_point_idx
        self.insertion_direction_idx_diff = insertion_direction_idx_diff
        self.insertion_vessel_name = insertion_vessel_name.lower()
        # 分支半径近似值（用于分叉检测）
        self.approx_branch_radii = approx_branch_radii
        # 可选旋转
        self.rotate_yzx_deg = rotate_yzx_deg
        # 是否检查中心线点是否在体网格内
        self.check_if_points_in_mesh = check_if_points_in_mesh
        print(f'self.check_if_points_in_mesh={self.check_if_points_in_mesh}')

        # 下载 / 定位模型目录
        self.model_folder = model_folder
        self.mesh_folder = os.path.join(self.model_folder, "Meshes")

        # 预读分支以初始化坐标空间
        branches = self._read_branches()
        self.coordinate_space = self._calc_coord_space(branches)
        self.coordinate_space_episode = self.coordinate_space

        # 运行时真正使用的结构（延迟初始化）
        self.branches = None
        self.insertion = None
        self.branching_points = None
        self.centerline_coordinates = None

        self._mesh_path = None

    # =========================
    # Mesh 路径接口
    # =========================
    @property
    def mesh_path(self) -> str:
        if self._mesh_path is None:
            self._make_mesh_obj()
        return self._mesh_path

    @property
    def visu_mesh_path(self) -> str:
        return self.mesh_path

    # =========================
    # Episode reset
    # =========================
    def reset(self, episode_nr=0, seed: int = None) -> None:
        if self.branches is None:
            self._make_branches()
        # print(f'branch after reset: {self.branches}')

    # =========================
    # 构建血管树结构
    # =========================
    def _make_branches(self):
        branches = self._read_branches()
        self.branches = branches

        self.coordinate_space = self._calc_coord_space(branches)
        centerline_coordinates = [branch.coordinates for branch in branches]
        self.centerline_coordinates = np.concatenate(centerline_coordinates)

        insert_vessel = self[self.insertion_vessel_name]
        ip, ip_dir = calc_insertion(
            insert_vessel,
            self.insertion_point_idx,
            self.insertion_point_idx + self.insertion_direction_idx_diff,
        )
        self.insertion = Insertion(ip, ip_dir)
        self.branching_points = calc_branching(self.branches, self.approx_branch_radii)
        self._mesh_path = None

    def _read_branches(self):
        mesh_path = _get_vtk_file(self.mesh_folder, ".vtu")
        # print(f'mesh_path = {mesh_path}')
        mesh = pv.read(mesh_path)
        mesh.scale([SCALING_FACTOR, SCALING_FACTOR, SCALING_FACTOR], inplace=True)
        branches = _get_branches(self.model_folder, mesh, self.check_if_points_in_mesh)

        if self.rotate_yzx_deg is not None:
            branches = rotate_branches(branches, self.rotate_yzx_deg)
        return branches

    def _calc_coord_space(self, branches):
        branch_highs = [branch.high for branch in branches]
        high = np.max(branch_highs, axis=0)
        branch_lows = [branch.low for branch in branches]
        low = np.min(branch_lows, axis=0)
        return gym.spaces.Box(low=low, high=high)

    def _make_mesh_obj(self):
        mesh_path = _get_vtk_file(self.mesh_folder, ".vtp")
        mesh = pv.read(mesh_path)
        mesh.flip_normals()
        mesh.scale([SCALING_FACTOR, SCALING_FACTOR, SCALING_FACTOR], inplace=True)
        if self.rotate_yzx_deg is not None:
            mesh.rotate_y(self.rotate_yzx_deg[0], inplace=True)
            mesh.rotate_z(self.rotate_yzx_deg[1], inplace=True)
            mesh.rotate_x(self.rotate_yzx_deg[2], inplace=True)
        mesh.decimate(0.9, inplace=True)

        obj_mesh_path = get_temp_mesh_path("VMR")
        pv.save_meshio(obj_mesh_path, mesh)
        self._mesh_path = obj_mesh_path
