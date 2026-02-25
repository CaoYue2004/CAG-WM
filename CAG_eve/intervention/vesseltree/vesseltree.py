from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mplot3d

# Branch / BranchingPoint：血管树结构中的“分支”与“分叉点”
from .util.branch import Branch, BranchingPoint
# EveObject：框架基类（通常提供配置构建、通用接口等）
from ...util import EveObject


# 用 dataclass 定义插入点信息：位置 + 方向
@dataclass
class Insertion:
    position: np.ndarray    # 插入点坐标 (3,)
    direction: np.ndarray   # 插入方向向量 (3,)（通常单位化）


# VesselTree：血管树抽象基类（提供分支/分叉点/中心线/插入点/坐标空间等）
class VesselTree(EveObject, ABC):
    # Set in subclasses in __init__() or reset():
    # ---- 下面这些字段由子类在 __init__() 或 reset() 中设置 ----
    branches: Tuple[Branch]                     # 所有分支（每个分支含坐标序列）
    branching_points: List[BranchingPoint]      # 所有分叉点（坐标+半径等）
    centerline_coordinates: np.ndarray          # 全局中心线点集（可能把所有分支拼起来）
    insertion: Insertion                        # 插入点（位置+方向）
    coordinate_space: gym.spaces.Box            # 全局坐标范围 Box(low, high)
    coordinate_space_episode: gym.spaces.Box    # 当前 episode 的坐标范围（可能更紧）
    mesh_path: str                              # 血管 mesh 文件路径（用于仿真/碰撞）
    visu_mesh_path: Optional[str] = None        # 可视化 mesh 路径（可选）

    def step(self) -> None:
        ...

    @abstractmethod
    def reset(self, episode_nr=0, seed: int = None) -> None:
        ...

    # 让 VesselTree 像 dict 一样用索引访问分支：
    # vesseltree[0] -> 第 0 条 branch；vesseltree["LAD"] -> 名字为 LAD 的 branch
    def __getitem__(self, item: Union[int, str]):
        # 如果传入 int，直接当作下标
        if isinstance(item, int):
            idx = item
        # 如果传入 str，则在 branch.name 列表里找到对应名字的下标
        else:
            branch_names = tuple(branch.name for branch in self.branches)
            idx = branch_names.index(item)
        # 返回对应分支对象
        return self.branches[idx]

    # 返回所有分支（值）
    def values(self) -> Tuple[Branch]:
        return self.branches

    # 返回所有分支名（键）
    def keys(self) -> Tuple[str]:
        return tuple(branch.name for branch in self.branches)

    # 返回 (name, branch) 的迭代器（类似 dict.items()）
    def items(self):
        branch_names = tuple(branch.name for branch in self.branches)
        return zip(branch_names, self.branches)

    def get_reset_state(self) -> Dict[str, Any]:
        state = {
            "branches": self.branches,
            "branching_points": self.branching_points,
            "centerline_coordinates": self.centerline_coordinates,
            "insertion": self.insertion,
            "coordinate_space": self.coordinate_space,
            "coordinate_space_episode": self.coordinate_space_episode,
        }
        return deepcopy(state)

    def get_step_state(self) -> Dict[str, Any]:
        state = {}
        return state


# 找到“离 point 最近”的分支（按分支坐标点到 point 的最小距离）
def find_nearest_branch_to_point(
    point: np.ndarray,      # 输入点 (3,)
    vessel_tree: VesselTree,    # 血管树
) -> Branch:
    nearest_branch = None   # 当前最接近的分支
    min_dist = np.inf       # 当前最小距离（初始化为无穷大）
    # 遍历所有分支
    for branch in vessel_tree.branches:
        # 计算该分支每个点到 point 的欧氏距离：shape (N,)
        distances = np.linalg.norm(branch.coordinates - point, axis=1)
        # 该分支的最小距离
        dist = np.min(distances)
        # 如果更近，则更新最优分支
        if dist < min_dist:
            min_dist = dist
            nearest_branch = branch
    return nearest_branch


# 判断 point 是否位于“血管树末端的开放端”（用于 stop_device_at_tree_end）
def at_tree_end(
    point: np.ndarray,
    vessel_tree: VesselTree,
) -> bool:
    # 如果 branches 还没初始化，直接认为不在末端
    if vessel_tree.branches is None:
        return False
    # 找到离 point 最近的分支
    branch = find_nearest_branch_to_point(point, vessel_tree)
    # 取该分支的坐标点数组 shape (N,3)
    branch_np = branch.coordinates
    # 计算 point 到该分支每个点的距离 shape (N,)
    distances = np.linalg.norm(branch_np - point, axis=1)
    # 找到最小距离对应的索引（最近的中心线点）
    min_idx = np.argmin(distances)
    # 找到“第二小距离”的索引（用 argpartition 快速找第 2 小）
    sec_min_idx = np.argpartition(distances, 1)[1]
    # 最近点 -> 次近点 的方向向量（近似局部切向）
    min_to_sec_min = branch_np[sec_min_idx] - branch_np[min_idx]
    # 最近点 -> point 的向量
    min_to_point = point - branch_np[min_idx]
    # 点积：用于判断 point 在最近点的“延伸方向”还是“反方向”
    dot_prod = np.dot(min_to_sec_min, min_to_point)

    # 条件1：最近点恰好是分支的端点（索引为 0 或 N-1）
    # 条件2：dot_prod <= 0 表示 point 在端点的“外侧/反向”，更像是越过端点了
    if (min_idx == 0 or min_idx == branch_np.shape[0] - 1) and dot_prod <= 0:
        # 取端点的坐标
        branch_point = branch.coordinates[min_idx]
        # 默认认为该端点是“开放端”（可以从这里出去）
        end_is_open = True

        # 遍历所有分叉点：如果端点落在某个分叉点半径范围内
        # 说明这里其实是“连接到其它分支”的端点，不是开放末端
        for branching_point in vessel_tree.branching_points:
            # 端点到分叉点中心的距离
            dist = np.linalg.norm(branching_point.coordinates - branch_point)
            # 若距离 < 分叉点半径，说明端点在分叉区域内 -> 不是开放末端
            if dist < branching_point.radius:
                end_is_open = False
        return end_is_open
    else:
        return False


# 画出血管树所有分支与分叉点（matplotlib 3D 调试用）
def plot_branches(vesseltree: VesselTree):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    coord_space = vesseltree.coordinate_space_episode
    margins = [
        (coord_space.high[0] - coord_space.low[0]),
        (coord_space.high[1] - coord_space.low[1]),
        (coord_space.high[2] - coord_space.low[2]),
    ]
    margin = max(margins)
    ax.set_xlim3d(
        coord_space.low[0],
        coord_space.low[0] + margin,
    )
    ax.set_ylim3d(
        coord_space.low[1],
        coord_space.low[1] + margin,
    )
    ax.set_zlim3d(
        coord_space.low[2],
        coord_space.low[2] + margin,
    )
    pick_event_handler = PickEventHandler(vesseltree)
    fig.canvas.mpl_connect("pick_event", pick_event_handler.on_click)
    for branch in vesseltree.branches:
        x = np.delete(branch.coordinates, [1, 2], axis=1).reshape(
            -1,
        )
        y = np.delete(branch.coordinates, [0, 2], axis=1).reshape(
            -1,
        )
        z = np.delete(branch.coordinates, [0, 1], axis=1).reshape(
            -1,
        )
        # ax.plot3D(x, y, z)
        line = mplot3d.art3d.Line3D(
            x, y, z, picker=True, pickradius=3, label=branch.name, color="red"
        )
        ax.add_artist(line)

    x = [
        branching_point.coordinates[0]
        for branching_point in vesseltree.branching_points
    ]
    y = [
        branching_point.coordinates[1]
        for branching_point in vesseltree.branching_points
    ]
    z = [
        branching_point.coordinates[2]
        for branching_point in vesseltree.branching_points
    ]

    bp_artist = mplot3d.art3d.Line3D(
        x, y, z, marker="o", color="g", markersize=8, linestyle="None"
    )
    ax.add_artist(bp_artist)
    # fig.canvas.draw()
    # plt.pause(0.001)
    # fig.canvas.start_event_loop(0.00001)
    plt.show()


# PickEventHandler：用于处理 plot_branches 的点击事件
class PickEventHandler:
    def __init__(self, vessel_tree: VesselTree) -> None:
        self.vessel_tree = vessel_tree

    def on_click(self, event):
        data = event.artist.get_data_3d()
        idx = event.ind[0] + int((event.ind[-1] - event.ind[0]) / 2)
        x = data[0][idx]
        y = data[1][idx]
        z = data[2][idx]
        point = np.array([x, y, z])
        branch = find_nearest_branch_to_point(point, self.vessel_tree)

        dist_to_point = np.linalg.norm(branch.coordinates - point, axis=1)
        idx = np.argmin(dist_to_point)

        print(branch.name)
        print(point)
        print(f"{idx=}")
