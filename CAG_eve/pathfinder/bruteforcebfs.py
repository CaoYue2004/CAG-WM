from typing import Dict, Generator, List, Tuple, NamedTuple
from copy import deepcopy
from math import inf
import numpy as np

from .pathfinder import Pathfinder
from ..intervention.vesseltree import (
    Branch,             # 一条血管分支（包含中心线坐标等）
    BranchingPoint,     # 分叉点（连接多条分支）
    find_nearest_branch_to_point,       # 给定点，找最近的分支
)

from ..intervention import Intervention
from ..util.coordtransform import tracking3d_to_vessel_cs, vessel_cs_to_tracking3d


# ---------------------------------------------------------------------
# 计算一条折线（path）的长度：相邻点距离求和
# ---------------------------------------------------------------------
def get_length(path: np.ndarray):
    return np.sum(np.linalg.norm(path[:-1] - path[1:], axis=1))


# ---------------------------------------------------------------------
# BPConnection：分叉点之间“沿某条分支走”的连接信息
# - length：连接路径长度
# - points：沿分支从 A 到 B 的路径点序列
# ---------------------------------------------------------------------
class BPConnection(NamedTuple):
    length: float
    points: np.ndarray


# ---------------------------------------------------------------------
# BruteForceBFS：通过 BFS 枚举图上的路径（从 start 到 target）
# 用于找到起始分支到目标分支之间的中心线路径
# ---------------------------------------------------------------------
class BruteForceBFS(Pathfinder):
    # ---------------------------------------------------------------
    # 构造：绑定 intervention，并初始化缓存
    # ---------------------------------------------------------------
    def __init__(self, intervention: Intervention):
        self.intervention = intervention
        self.path_length: float = 0.0
        self.path_points3d: np.ndarray = np.empty((0, 3))
        self.path_branching_points3d: np.ndarray = np.empty((0, 3))
        self._branches = None
        self._node_connections = None
        self._search_graph_base = None

    # ---------------------------------------------------------------
    # reset：如果血管树发生变化则重新初始化图结构，然后计算一次路径
    # ---------------------------------------------------------------
    def reset(self, episode_nr=0) -> None:
        if self._branches != self.intervention.vessel_tree.branches:
            self._init_vessel_tree()
            self.path_length = 0.0
            self.path_points3d = np.empty((0, 3))
            self.path_branching_points3d = np.empty((0, 3))
            self._branches = self.intervention.vessel_tree.branches
        self.step()

    # ---------------------------------------------------------------
    # step：读取当前位置与目标点，计算最短中心线路径（并输出 tracking3d 坐标）
    # ---------------------------------------------------------------
    def step(self) -> None:
        fluoro = self.intervention.fluoroscopy  # 取 fluoroscopy（提供 tracking3d 与坐标变换参数）
        position = fluoro.tracking3d[0]     # 当前位置：tracking3d 的第 0 个点（通常是器械尖端）
        position_vessel_cs = tracking3d_to_vessel_cs(       # 将当前位置从 tracking 坐标系转换到 vessel 坐标系
            position, fluoro.image_rot_zx, fluoro.image_center
        )
        target = self.intervention.target.coordinates3d     # 取目标点（tracking 坐标系）
        target_vessel_cs = tracking3d_to_vessel_cs(         # 将目标点转换到 vessel 坐标系
            target, fluoro.image_rot_zx, fluoro.image_center
        )
        position_branch = find_nearest_branch_to_point(     # 找“当前位置”最近的分支
            position_vessel_cs, self.intervention.vessel_tree
        )
        target_branch = find_nearest_branch_to_point(       # 找“目标点”最近的分支
            target_vessel_cs, self.intervention.vessel_tree
        )

        # 计算最短路径：
        # - path_branching_points：经过哪些分叉点（对象）
        # - self.path_length：总长度
        # - path_points：vessel 坐标系的路径点序列
        (
            path_branching_points,
            self.path_length,
            path_points,
        ) = self._get_shortest_path(
            position_branch, target_branch, position_vessel_cs, target_vessel_cs
        )
        # 如果路径中包含分叉点列表（非 None）
        if path_branching_points is not None:
            # 取出每个分叉点的坐标
            path_branching_points = [
                branching_point.coordinates for branching_point in path_branching_points
            ]
            path_branching_points = np.array(path_branching_points)
            # 将分叉点坐标从 vessel 坐标系变回 tracking3d
            self.path_branching_points3d = vessel_cs_to_tracking3d(
                path_branching_points,
                fluoro.image_rot_zx,
                fluoro.image_center,
                fluoro.field_of_view,
            )
        # 如果没有分叉点（例如同分支直连 or 无路径）
        else:
            self.path_branching_points3d = None
        # 将完整路径点从 vessel 坐标系变回 tracking3d
        self.path_points3d = vessel_cs_to_tracking3d(
            path_points,
            fluoro.image_rot_zx,
            fluoro.image_center,
            fluoro.field_of_view,
        )

    # ---------------------------------------------------------------
    # 初始化血管树的图结构：分叉点连接表 + 基础搜索图
    # ---------------------------------------------------------------
    def _init_vessel_tree(self) -> None:
        self._node_connections = self._initialize_node_connections(
            self.intervention.vessel_tree.branching_points
        )
        # 建立 BFS 的基础图结构（邻接表）
        self._search_graph_base = self._initialize_search_graph_base()

    # ---------------------------------------------------------------
    # 建立分叉点之间的连接字典
    # 返回结构：
    # node_connections[bp1][bp2] = BPConnection(length, points)
    # ---------------------------------------------------------------
    def _initialize_node_connections(
        self, branching_points: Tuple[BranchingPoint]
    ) -> Dict[BranchingPoint, Dict[BranchingPoint, BPConnection]]:
        node_connections = {}
        # 遍历每个分叉点
        for branching_point in branching_points:
            node_connections[branching_point] = {}
            # 遍历该分叉点连接到的分支（connection 是 Branch）
            for connection in branching_point.connections:
                # 遍历所有分叉点，找“共享同一条分支”的另一个分叉点
                for target_branching_point in branching_points:
                    if branching_point == target_branching_point:
                        continue
                    # 如果 target_branching_point 也连接了这条分支，则两者可通过该分支互达
                    if connection in target_branching_point.connections:
                        # 计算沿该分支从 bp -> target_bp 的路径点序列
                        points = connection.get_path_along_branch(
                            branching_point.coordinates,
                            target_branching_point.coordinates,
                        )
                        # 计算这段路径的长度
                        length = get_length(points)

                        # 写入连接表：bp 到 target_bp 的连接信息
                        node_connections[branching_point][
                            target_branching_point
                        ] = BPConnection(length, points)
        return node_connections

    # ---------------------------------------------------------------
    # 基于 node_connections 生成 BFS 的基础邻接表
    # _search_graph_base[node] = [neighbor1, neighbor2, ...]
    # ---------------------------------------------------------------
    def _initialize_search_graph_base(
        self,
    ) -> Dict[BranchingPoint, List[BranchingPoint]]:
        _search_graph_base = {}
        for node in self._node_connections:
            _search_graph_base[node] = list(self._node_connections[node].keys())
        return _search_graph_base

    # ---------------------------------------------------------------
    # 计算最短路径：
    # start_branch / target_branch：起点所在分支、目标所在分支
    # start / target：起点与目标点在 vessel_cs 下的坐标
    # 返回：
    # - shortest_path：经过的分叉点列表（不含 start/target 字符串节点）
    # - shortest_path_length：总长度
    # - shortest_path_points：完整路径点序列（vessel_cs）
    # ---------------------------------------------------------------
    def _get_shortest_path(
        self,
        start_branch: Branch,
        target_branch: Branch,
        start: np.ndarray,
        target: np.ndarray,
    ):  # -> Tuple[List[BranchingPoint], float, List[CenterlinePoint]]:
        # 构建当前任务的搜索图（加入 'start' 与 'target' 虚拟节点）
        search_graph = self._create_search_graph(start_branch, target_branch)
        # 获得 BFS 的路径生成器（会依次 yield 从 start 到 target 的路径）
        bfs_paths = self._get_bfs_paths_generator(search_graph)

        shortest_path_length = inf
        shortest_path = None

        # 取 BFS 的第一条路径（BFS 的第一条路径就是“最少边数”的路径）
        path = next(bfs_paths, None)
        # 如果没有任何路径（极端情况：图不连通）
        if path is None:
            shortest_path_points = np.empty((1, 3))
            shortest_path_length = 0.0

        # 如果路径长度为 2：['start', 'target']，说明起点分支与目标分支相同（或直接连接）
        elif len(path) == 2:
            # 直接在 start_branch 上取从 start 到 target 的路径点
            shortest_path_points = start_branch.get_path_along_branch(start, target)
            shortest_path_length = get_length(shortest_path_points)

        # 否则：路径会形如 ['start', bp1, bp2, ..., 'target']
        else:
            # 先取从起点到第一个分叉点（bp1）的路径点
            shortest_path_points = start_branch.get_path_along_branch(
                start, path[1].coordinates
            )
            # 累加这段长度
            shortest_path_length = get_length(shortest_path_points)

            # 遍历中间分叉点对（bp1->bp2, bp2->bp3, ...），不包含最后连接到 target 的段
            for node, next_node in zip(path[1:-2], path[2:-1]):
                connection = self._node_connections[node][next_node]
                shortest_path_length += connection.length
                shortest_path_points = np.vstack(
                    (shortest_path_points, connection.points[1:])
                )

            target_points = target_branch.get_path_along_branch(
                path[-2].coordinates, target
            )

            target_length = get_length(target_points)
            shortest_path_points = np.vstack((shortest_path_points, target_points[1:]))
            shortest_path_length += target_length
            shortest_path = path[1:-1]

        # 返回：分叉点序列、总长度、完整路径点
        return shortest_path, shortest_path_length, shortest_path_points

    # ---------------------------------------------------------------
    # 构建带虚拟节点的搜索图：
    # - 添加 'start' 节点，连到起始分支所在的所有分叉点
    # - 将所有“与 target_branch 相连”的分叉点额外连到 'target'
    # ---------------------------------------------------------------
    def _create_search_graph(self, start_branch, target_branch):
        search_graph = deepcopy(self._search_graph_base)
        # 如果起始分支与目标分支相同，直接 start -> target
        if start_branch == target_branch:
            search_graph["start"] = ["target"]
            return search_graph

        start_connections = []
        for branching_point in self.intervention.vessel_tree.branching_points:
            # 如果该分叉点连接了 start_branch，则作为 start 的邻居
            if start_branch in branching_point.connections:
                start_connections.append(branching_point)
            # 如果该分叉点连接了 target_branch，则该分叉点可以直接到达 'target'
            if target_branch in branching_point.connections:
                search_graph[branching_point].append("target")

        # 设置虚拟 start 节点的邻居
        search_graph["start"] = start_connections

        return search_graph

    # ---------------------------------------------------------------
    # BFS 路径生成器：从 'start' 到 'target' 枚举路径
    # graph 是邻接表，节点既可能是 BranchingPoint，也可能是字符串 'start'/'target'
    # ---------------------------------------------------------------
    def _get_bfs_paths_generator(
        self, graph: Dict
    ) -> Generator[List[BranchingPoint], None, None]:
        """bfs path search

        Arguments:
            graph {dict} -- dict of nodes containing all connected nodes
                including the entries 'start' and 'target'

        Yields:
            [list] -- a list of the nodes along the path with the node
                names as entries
        """
        queue = [("start", ["start"])]
        while queue:
            (vertex, path) = queue.pop(0)
            for next_bp in graph[vertex]:
                if next_bp in path:
                    continue
                elif next_bp == "target":
                    yield path + [next_bp]
                else:
                    queue.append((next_bp, path + [next_bp]))
