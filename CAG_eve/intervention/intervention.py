# pylint: disable=unused-argument
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional
import gymnasium as gym

import numpy as np
from ..util import EveObject
# Target：目标模块（导航目标/终点）
from .target import Target
# VesselTree：血管树模块（mesh、中心线、插入点等）
from .vesseltree import VesselTree
# Fluoroscopy：透视成像模块；SimulatedFluoroscopy：基于仿真的透视实现
from .fluoroscopy import Fluoroscopy, SimulatedFluoroscopy
# Simulation：仿真接口；SimulationMP：多进程/隔离进程的仿真封装（提高稳定性/避免卡死）
from .simulation import Simulation, SimulationMP
# Device：器械模块（导丝/导管的形状参数、tip等）
from .device import Device
from .navigator import Navigator


# =========================
# Intervention：介入过程抽象基类（核心接口）
# =========================
class Intervention(EveObject, ABC):
    # ---- 下面这些是“类型注解的类属性声明” ----
    # 注意：这里只是声明“这个类/子类会有这些字段”，不是在这里赋值创建对象
    devices: List[Device]       # 介入中可能有多个器械（导丝+导管等）
    vessel_tree: VesselTree     # 血管树对象
    fluoroscopy: Fluoroscopy    # 成像/反馈模块
    target: Target              # 目标对象（终点/导航目标）
    simulation: Simulation      # 仿真
    navigator: Navigator        # 路径导航
    normalize_action: bool = True      # 是否把 action 归一化（默认不归一化）
    last_action: np.ndarray             # 记录上一步动作（用于 reward/info/debug）
    device_lengths_inserted: List[float]    # 每个器械推进的“插入长度”
    device_rotations: List[float]           # 每个器械当前旋转量/角度
    device_lengths_maximum: List[float]     # 每个器械允许的最大插入长度（约束）
    device_diameters: List[float]           # 每个器械允许的最大插入长度（约束）
    action_space: gym.spaces.Box            # Gym 动作空间（通常是连续 Box）

    @abstractmethod
    def step(self, action: np.ndarray) -> None:
        ...

    @abstractmethod
    def reset(
        self,
        episode_number: int,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    @abstractmethod
    def reset_devices(self) -> None:
        ...

    # ---- 状态快照：reset 后的完整状态（给 debug/记录/回放用） ----
    def get_reset_state(self) -> Dict[str, Any]:
        state = {
            "devices": self.devices,
            "device_lengths_inserted": self.device_lengths_inserted,
            "device_rotations": self.device_rotations,
            "device_lengths_maximum": self.device_lengths_maximum,
            "device_diameters": self.device_diameters,
            "action_space": self.action_space,
            "last_action": self.last_action,
            "vessel_tree": self.vessel_tree.get_reset_state(),
            "target": self.target.get_reset_state(),
            "fluoroscopy": self.fluoroscopy.get_reset_state(),
            "navigator_path": self.navigator.route_pts_local
        }
        return deepcopy(state)

    # ---- 状态快照：step 后的状态（更轻量，适合每步记录） ----
    def get_step_state(self) -> Dict[str, Any]:
        state = {
            "device_lengths_inserted": self.device_lengths_inserted,
            "device_rotations": self.device_rotations,
            "last_action": self.last_action,
            "vessel_tree": self.vessel_tree.get_step_state(),
            "target": self.target.get_step_state(),
            "fluoroscopy": self.fluoroscopy.get_step_state(),
            "navigator_path": self.navigator.route_pts_local
        }
        return deepcopy(state)


# =========================
# SimulatedIntervention：带物理仿真的 Intervention 抽象基类
# =========================
class SimulatedIntervention(Intervention, ABC):
    # 这里进一步收窄类型：fluoroscopy 必须是 SimulatedFluoroscopy（依赖 simulation）
    fluoroscopy: SimulatedFluoroscopy
    # 这个类比 Intervention 多了一个 simulation（物理/几何仿真后端）
    simulation: Simulation
    # 一个逻辑开关：器械到达血管末端时是否强制停住（避免穿出血管树）
    stop_device_at_tree_end: bool = True

    # 把仿真切换成多进程版本（MP），防止 step 卡死、隔离崩溃
    def make_mp(self, step_timeout: float = 2, restart_n_resets: int = 200):
        # 如果当前 simulation 是普通 Simulation（非 MP 包装）
        if isinstance(self.simulation, Simulation):
            # 用 SimulationMP 包一层
            # step_timeout：每步最大允许时间（秒），超时可重启/报错
            # restart_n_resets：reset 多少次后重启一次进程（防内存泄漏/累积崩溃）
            new_sim = SimulationMP(self.simulation, step_timeout, restart_n_resets)
            self.fluoroscopy.simulation = new_sim
            self.simulation = new_sim

    # 从多进程版本切回普通版本
    def make_non_mp(self):
        if isinstance(self.simulation, SimulationMP):
            new_sim = self.simulation.simulation
            self.fluoroscopy.simulation = new_sim
            self.simulation = new_sim

    def get_reset_state(self) -> Dict[str, Any]:
        state = {
            "devices": self.devices,
            "device_lengths_inserted": self.device_lengths_inserted,
            "device_rotations": self.device_rotations,
            "device_lengths_maximum": self.device_lengths_maximum,
            "device_diameters": self.device_diameters,
            "action_space": self.action_space,
            "last_action": self.last_action,
            "vessel_tree": self.vessel_tree.get_reset_state(),
            "simulation": self.simulation.get_reset_state(),
            "target": self.target.get_reset_state(),
            "fluoroscopy": self.fluoroscopy.get_reset_state(),
            "navigator_path": self.navigator.route_pts_local
        }
        return deepcopy(state)

    def get_step_state(self) -> Dict[str, Any]:
        state = {
            "device_lengths_inserted": self.device_lengths_inserted,
            "device_rotations": self.device_rotations,
            "last_action": self.last_action,
            "vessel_tree": self.vessel_tree.get_step_state(),
            "simulation": self.simulation.get_step_state(),
            "target": self.target.get_step_state(),
            "fluoroscopy": self.fluoroscopy.get_step_state(),
            "navigator_path": self.navigator.route_pts_local
        }
        return deepcopy(state)
