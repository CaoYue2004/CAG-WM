from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import heapq

from .navigator import Navigator
from ..vesseltree import VesselTree
from ..target import Target
from ..simulation import Simulation


class VesselGraph:
    def __init__(self, round_ndigits: int = 6):
        self.round_ndigits = round_ndigits
        self.nodes: List[np.ndarray] = []  # List[(3,)]
        self.adj: List[List[Tuple[int, float]]] = []
        self._key2nid: Dict[Tuple[float, float, float], int] = {}

    def _key(self, p: np.ndarray) -> Tuple[float, float, float]:
        return (
            round(float(p[0]), self.round_ndigits),
            round(float(p[1]), self.round_ndigits),
            round(float(p[2]), self.round_ndigits),
        )

    def _get_or_add_node(self, p: np.ndarray) -> int:
        k = self._key(p)    
        nid = self._key2nid.get(k)      
        if nid is None:     
            nid = len(self.nodes)       
            self._key2nid[k] = nid      
            self.nodes.append(np.asarray(p, dtype=np.float32))      
            self.adj.append([])     
        return nid

    def add_undirected_edge(self, u: int, v: int, w: float) -> None:
        self.adj[u].append((v, w))      
        self.adj[v].append((u, w))      

    @classmethod
    def from_branches(cls, branches, round_ndigits: int = 6) -> "VesselGraph":
        """
        branches: List[Branch], Branch.coordinates: (M,3)
        """
        g = cls(round_ndigits=round_ndigits)
        for br in branches:    
            coords = np.asarray(br.coordinates, dtype=np.float32)       
            if coords.ndim != 2 or coords.shape[0] < 2:
                continue
            prev = g._get_or_add_node(coords[0])
            for i in range(1, coords.shape[0]):         
                cur = g._get_or_add_node(coords[i])     
                if cur != prev:     
                    w = float(np.linalg.norm(coords[i] - coords[i - 1]))       
                    if w > 1e-6:
                        g.add_undirected_edge(prev, cur, w)
                prev = cur      
        return g

    def nearest_node(self, p: np.ndarray) -> int:
        p = np.asarray(p, dtype=np.float32).reshape(3,)
        k = self._key(p)
        if k in self._key2nid:      
            return int(self._key2nid[k])        

        X = np.stack(self.nodes, axis=0)  
        d2 = np.sum((X - p.reshape(1, 3)) ** 2, axis=1)
        return int(np.argmin(d2))       


class GraphNavigator(Navigator):

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
        self.k_route = int(k_route)     
        self.round_ndigits = round_ndigits
        self.graph = None

        self.dist_to_target: Optional[np.ndarray] = None      # (N,)
        self.next_hop: Optional[np.ndarray] = None            
        self.target_nid: int = -1

        self.target_point_world = np.zeros((3,), dtype=np.float32)
        self.dist_to_target_scalar = float("inf")
        self.route_pts_world = np.zeros((self.k_route, 3), dtype=np.float32)
        self.route_pts_local = np.zeros((self.k_route, 3), dtype=np.float32)

    def reset(self) -> None:
        if self.graph is None:
            branches = self.vessel_tree.branches
            self.graph = VesselGraph.from_branches(branches, round_ndigits=self.round_ndigits)

        self.target_point_world = np.asarray(self.target.coordinates3d, dtype=np.float32).reshape(3,)
        self.target_nid = self.graph.nearest_node(self.target_point_world)

        n = len(self.graph.nodes)
        dist = np.full((n,), np.inf, dtype=np.float32)
        parent = np.full((n,), -1, dtype=np.int32)

        dist[self.target_nid] = 0.0
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

        self.dist_to_target_scalar = 0.0
        self.route_pts_world[:] = self.graph.nodes[self.target_nid]
        self.route_pts_local[:] = 0.0

    def step(self) -> None:
        assert self.dist_to_target is not None and self.next_hop is not None, "GraphNavigator.reset() must be called first."

        tip_point_world = np.asarray(self.simulation.dof_positions[-1], dtype=np.float32).reshape(3,)
        v_tip = self.graph.nearest_node(tip_point_world)

        self.dist_to_target_scalar = float(self.dist_to_target[v_tip])

        route_ids: List[int] = []
        cur = v_tip
        for _ in range(self.k_route):
            route_ids.append(cur)
            nxt = int(self.next_hop[cur])
            if nxt < 0:
                break
            cur = nxt

        last = route_ids[-1]
        while len(route_ids) < self.k_route:
            route_ids.append(last)

        route_world = np.stack([self.graph.nodes[i] for i in route_ids], axis=0).astype(np.float32)
        # print(f'tip_point_world={tip_point_world}')
        # print(f'route_world={route_world}')

        self.route_pts_world = route_world

        self.route_pts_local = (route_world - tip_point_world).astype(np.float32)
        # print(f'route_pts_local={self.route_pts_local}')




