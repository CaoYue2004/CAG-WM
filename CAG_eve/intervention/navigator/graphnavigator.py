from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import heapq

from .navigator import Navigator
from ..vesseltree import VesselTree
from ..target import Target
from ..simulation import Simulation

# ============================================================
# VesselGraph：普通工具类（不继承 EveObject）
# ============================================================

class VesselGraph:
    """
    节点：中心线采样点 (3D)
    边：相邻采样点连边，权重为欧氏距离（近似弧长）
    分叉点完全共享 -> 用坐标 key 去重即可连通
    """

    def __init__(self, round_ndigits: int = 6):
        # 用于把 float 坐标 round 到固定精度，避免 float 表示误差导致“同一点”无法匹配
        self.round_ndigits = round_ndigits
        # nodes[nid] = 该节点在 3D 空间中的坐标 (x,y,z)，按加入顺序存
        self.nodes: List[np.ndarray] = []  # List[(3,)]
        # adj[u] = [(v, w), ...] 表示从节点 u 可以到达节点 v，边权为 w（距离/弧长）
        self.adj: List[List[Tuple[int, float]]] = []  # 邻接表: (neighbor, weight)
        # 把“坐标key”映射到节点id：
        # key = (round(x), round(y), round(z)) -> nid
        # 用于去重与快速查找（O(1)）
        self._key2nid: Dict[Tuple[float, float, float], int] = {}

    def _key(self, p: np.ndarray) -> Tuple[float, float, float]:
        # 把一个 3D 点坐标 p 映射为 hashable 的 key（tuple），用于字典索引
        return (
            round(float(p[0]), self.round_ndigits),
            round(float(p[1]), self.round_ndigits),
            round(float(p[2]), self.round_ndigits),
        )

    def _get_or_add_node(self, p: np.ndarray) -> int:
        # 给定一个点 p：
        # - 如果这个点（按 key）已经存在于图中，返回已有的节点 id
        # - 否则创建新节点，加入 nodes/adj，并返回新 id
        k = self._key(p)    # 计算坐标 key（用于去重）
        nid = self._key2nid.get(k)      # 看看这个 key 是否已经对应一个节点 id
        if nid is None:     # 不存在 -> 创建新节点
            nid = len(self.nodes)       # 新节点 id：当前节点数（append 前）
            self._key2nid[k] = nid      # 登记 key -> nid 映射
            self.nodes.append(np.asarray(p, dtype=np.float32))      # 把点坐标存到 nodes（统一 float32，省内存/对齐 torch）
            self.adj.append([])     # 为新节点创建一个空邻接表（后面会往里面 add edge）
        return nid

    def add_undirected_edge(self, u: int, v: int, w: float) -> None:
        # 在无向图中添加边 u <-> v，权重为 w
        self.adj[u].append((v, w))      # u 的邻居包含 v
        self.adj[v].append((u, w))      # v 的邻居包含 u（无向边）

    @classmethod
    def from_branches(cls, branches, round_ndigits: int = 6) -> "VesselGraph":
        """
        branches: List[Branch], Branch.coordinates: (M,3)
        """
        # 创建一个新图对象，round_ndigits 用于坐标去重
        g = cls(round_ndigits=round_ndigits)
        for br in branches:     # 创建一个新图对象，round_ndigits 用于坐标去重
            coords = np.asarray(br.coordinates, dtype=np.float32)       # 把该分支的坐标转成 (M,3) 的 numpy 数组
            if coords.ndim != 2 or coords.shape[0] < 2:
                continue

            # 取分支第一个点，获取/创建节点 id，作为 prev（上一节点）
            prev = g._get_or_add_node(coords[0])
            for i in range(1, coords.shape[0]):         # 从第二个点开始逐段连边
                cur = g._get_or_add_node(coords[i])     # 当前点获取/创建节点 id
                if cur != prev:     # 如果当前点和前一个点在 key 去重后不是同一个节点
                    w = float(np.linalg.norm(coords[i] - coords[i - 1]))        # 用原始坐标计算相邻采样点的欧氏距离，作为边权（近似弧长）
                    if w > 1e-6:
                        g.add_undirected_edge(prev, cur, w)
                prev = cur      # 更新 prev，准备连接下一段
        return g

    def nearest_node(self, p: np.ndarray) -> int:
        """
        - 若 p 恰好是中心线点：key 命中 O(1)
        - 否则：暴力最近邻 O(N)（点多时可替换 KDTree）
        """
        p = np.asarray(p, dtype=np.float32).reshape(3,)
        k = self._key(p)
        if k in self._key2nid:      # 如果 p 恰好落在中心线采样点（或同一 round 精度）
            return int(self._key2nid[k])        # O(1) 返回对应节点 id

        X = np.stack(self.nodes, axis=0)  # (N,3)
        d2 = np.sum((X - p.reshape(1, 3)) ** 2, axis=1)
        return int(np.argmin(d2))       # 返回距离最小的节点 id（最近邻）


# ============================================================
# GraphNavigator：具体实现（继承 Navigator/EveObject）
# ============================================================

class GraphNavigator(Navigator):
    """
    reset: 从 target 跑一次 Dijkstra，得到 dist_to_target / next_hop
    step : 根据 tip 最近节点，沿 next_hop 走 k_route 得到 route
    作用：在 reset 时构建“导航表”，在 step 时查询导航表生成路径提示（route）
    """

    def __init__(
        self,
        vessel_tree: VesselTree,
        target: Target,
        simulation: Simulation,
        k_route: int = 32,
        round_ndigits: int = 6,
    ) -> None:
        self.vessel_tree = vessel_tree
        self.target = target
        self.simulation = simulation
        self.k_route = int(k_route)     # 每步输出的路径点数量 K（固定长度，便于作为神经网络输入）
        self.round_ndigits = round_ndigits
        # 用 vessel_tree.brunches（每条 branch 是中心线 polyline）构一张图：
        # 节点=中心线采样点，边=相邻采样点，权重=距离（近似弧长）
        # 分叉点共享 -> 会自动连通
        # branches = self.vessel_tree.branches
        # self.graph = VesselGraph.from_branches(branches, round_ndigits=self.round_ndigits)
        self.graph = None

        # 导航表（reset 后有效）
        # 从节点 v 到 target 的最短路距离
        self.dist_to_target: Optional[np.ndarray] = None      # (N,)
        # 从节点 v 出发，朝 target 走的“下一跳节点”
        self.next_hop: Optional[np.ndarray] = None            # (N,) int, v -> 下一跳
        # target 对应的图节点 id（reset 时由 target_point_world 投影得到）
        self.target_nid: int = -1

        # 输出缓存（固定 shape）
        # 存当前 episode 的 target 3D 坐标（世界/tracking 坐标系）
        self.target_point_world = np.zeros((3,), dtype=np.float32)
        # 当前 tip 到 target 的最短路距离（标量），step 时更新
        self.dist_to_target_scalar = float("inf")
        # 当前 step 的“前方路径点”（世界坐标），shape 固定 (K,3)
        self.route_pts_world = np.zeros((self.k_route, 3), dtype=np.float32)
        # 当前 step 的“前方路径点”（tip 局部坐标），shape 固定 (K,3)
        self.route_pts_local = np.zeros((self.k_route, 3), dtype=np.float32)

    def reset(self) -> None:
        if self.graph is None:
            branches = self.vessel_tree.branches
            self.graph = VesselGraph.from_branches(branches, round_ndigits=self.round_ndigits)

        self.target_point_world = np.asarray(self.target.coordinates3d, dtype=np.float32).reshape(3,)
        # 把 target 3D 点“投影”到图上：找到最近的图节点作为 target 节点 id
        self.target_nid = self.graph.nearest_node(self.target_point_world)

        # 图节点总数 N
        n = len(self.graph.nodes)
        # dist[v] 初始化为无穷大：表示尚未找到从 v 到 target 的最短距离
        dist = np.full((n,), np.inf, dtype=np.float32)
        # 所以 parent[v] 正好可以作为 v 走向 target 的 next_hop[v]
        parent = np.full((n,), -1, dtype=np.int32)

        # target 自己到 target 的距离为 0（Dijkstra 的源点）
        dist[self.target_nid] = 0.0
        # 优先队列（小根堆）：(当前最短距离, 节点id)
        # 初始只有源点 target
        pq: List[Tuple[float, int]] = [(0.0, self.target_nid)]
        visited = np.zeros((n,), dtype=np.bool_)

        while pq:
            d_u, u = heapq.heappop(pq)
            if visited[u]:
                continue
            visited[u] = True
            for v, w in self.graph.adj[u]:
                nd = d_u + w
                if nd < dist[v]:
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(pq, (float(nd), v))

        self.dist_to_target = dist
        self.next_hop = parent

        # reset 时输出也初始化一下（可选）
        self.dist_to_target_scalar = 0.0
        self.route_pts_world[:] = self.graph.nodes[self.target_nid]
        self.route_pts_local[:] = 0.0

    def step(self) -> None:
        assert self.dist_to_target is not None and self.next_hop is not None, "GraphNavigator.reset() must be called first."

        tip_point_world = np.asarray(self.simulation.dof_positions[-1], dtype=np.float32).reshape(3,)
        # 把 tip 位置投影到图：找到最近的图节点 id 作为当前所在节点
        v_tip = self.graph.nearest_node(tip_point_world)

        # 1) dist 标量
        # 查表得到当前 tip 到 target 的最短路距离（用于奖励 shaping / 观测输入）
        self.dist_to_target_scalar = float(self.dist_to_target[v_tip])

        # 2) 沿 next_hop 走 k_route 步取路径点
        route_ids: List[int] = []
        cur = v_tip
        for _ in range(self.k_route):
            route_ids.append(cur)
            nxt = int(self.next_hop[cur])
            if nxt < 0:
                break
            cur = nxt

        # 补齐固定长度
        last = route_ids[-1]
        while len(route_ids) < self.k_route:
            route_ids.append(last)

        # 把节点 id 映射回 3D 坐标，得到 (K,3) 的世界坐标路径点
        route_world = np.stack([self.graph.nodes[i] for i in route_ids], axis=0).astype(np.float32)
        # print(f'tip_point_world={tip_point_world}')
        # print(f'route_world={route_world}')

        self.route_pts_world = route_world

        self.route_pts_local = (route_world - tip_point_world).astype(np.float32)
        # print(f'route_pts_local={self.route_pts_local}')



