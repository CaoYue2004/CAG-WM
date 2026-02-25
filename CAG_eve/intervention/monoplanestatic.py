# pylint: disable=unused-argument
from typing import Any, Dict, List, Optional
import gymnasium as gym

import numpy as np
# 导入“带仿真”的介入基类（比 Intervention 多一个 simulation，并提供 MP 切换等）
from .intervention import SimulatedIntervention
# 目标模块：提供导航目标、reset/step 状态等
from .target import Target
# 血管树模块：提供 mesh、插入点、坐标范围等
from .vesseltree import VesselTree
# at_tree_end：判断导丝/导管 tip 是否到达血管树末端（用于防止继续推进）
from .vesseltree.vesseltree import at_tree_end
# SimulatedFluoroscopy：基于 simulation 的透视成像（生成 x-ray 或 tracking）
from .fluoroscopy import SimulatedFluoroscopy
# Device：器械（导丝/导管）参数与 SOFA/仿真对象封装
from .device import Device
# Simulation：物理/几何仿真后端接口（真正执行 step(action,duration)）
from .simulation import Simulation
from .navigator import GraphNavigator


# 单平面 + 静态血管的“仿真介入”实现类
class MonoPlaneStatic(SimulatedIntervention):
    def __init__(
        self,
        vessel_tree: VesselTree,        # 血管树（mesh/中心线/插入点等）
        devices: List[Device],          # 器械列表（可能多个）
        simulation: Simulation,         # 仿真后端（SOFA/自定义等）
        fluoroscopy: SimulatedFluoroscopy,      # 透视成像模块（依赖 simulation）
        target: Target,         # 目标模块
        navigator: GraphNavigator,
        stop_device_at_tree_end: bool = True,       # tip 到末端是否停止推进
        normalize_action: bool = False,         # 动作是否归一化到 [-1,1]
    ) -> None:
        self.vessel_tree = vessel_tree
        self.devices = devices
        self.target = target
        self.navigator = navigator
        self.fluoroscopy = fluoroscopy
        self.stop_device_at_tree_end = stop_device_at_tree_end
        self.normlaize_action = normalize_action
        self.simulation = simulation
        # numpy 随机数生成器（用于分发子模块 seed）
        self._np_random = np.random.default_rng()

        # 从每个 device 取速度上限，组成数组
        # 常见形状是 (n_devices, 2)：两维可能对应 [推进速度, 旋转速度]
        self.velocity_limits = np.array(
            [device.velocity_limit for device in self.devices]
        )
        # last_action：记录上一帧动作（形状同 velocity_limits）
        self.last_action = np.zeros_like(self.velocity_limits)
        # 缓存 inserted_lengths/rotations 的引用（但下面 property 又直接读 simulation，通常不必缓存）
        self._device_lengths_inserted = self.simulation.inserted_lengths
        self._device_rotations = self.simulation.rotations
        self._last_simulation_error = False

    # 当前插入长度：直接从 simulation 读取（simulation 是“真源”）
    @property
    def device_lengths_inserted(self) -> List[float]:
        return self.simulation.inserted_lengths

    # 当前旋转量：直接从 simulation 读取
    @property
    def device_rotations(self) -> List[float]:
        return self.simulation.rotations

    # 每个器械最大允许插入长度：来自 device 参数
    @property
    def device_lengths_maximum(self) -> List[float]:
        return [device.length for device in self.devices]

    # 每个器械直径：来自 sofa_device 半径 * 2
    @property
    def device_diameters(self) -> List[float]:
        return [device.sofa_device.radius * 2 for device in self.devices]

    # 动作空间：根据 normalize_action 决定是 [-1,1] 还是 [-vmax,vmax]
    '''@property
    def action_space(self) -> gym.spaces.Box:
        # 如果使用归一化动作：每个维度范围 [-1,1]
        if self.normalize_action:
            high = np.ones_like(self.velocity_limits)
            space = gym.spaces.Box(low=-high, high=high)
        # 否则：每个维度范围 [-velocity_limits, velocity_limits]
        else:
            space = gym.spaces.Box(low=-self.velocity_limits, high=self.velocity_limits)
        return space'''

    @property
    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self.velocity_limits.shape,
            dtype=np.float32
        )

    # 推进一步仿真
    def step(self, action: np.ndarray) -> None:
        # 把 action 转成 numpy 并 reshape 成 (n_devices, 2)（或与 velocity_limits 同形状）
        action = np.array(action).reshape(self.velocity_limits.shape)
        # ---- 动作归一化模式：把 [-1,1] 映射回真实速度区间 ----
        if self.normalize_action:
            action = np.asarray(action, dtype=np.float32)
            action = np.clip(action, -1.0, 1.0)  # (B,2) 或 (2,)

            vel = np.asarray(self.velocity_limits, dtype=np.float32).reshape(-1)
            v_max, w_max = vel[0], vel[1]
            v_min = 1.0

            if action.ndim == 1:
                action = action[None, :]  # -> (1,2)

            v_norm = action[:, 0]  # (B,)
            w_norm = action[:, 1]  # (B,)

            v_real = np.zeros_like(v_norm)
            pos = v_norm > 0
            neg = v_norm < 0
            v_real[pos] = v_min + v_norm[pos] * (v_max - v_min)
            v_real[neg] = -v_min + v_norm[neg] * (v_max - v_min)

            w_real = w_norm * w_max

            action = np.stack([v_real, w_real], axis=1).astype(np.float32)  # (B,2)
            # print(f'action={action}')
        # ---- 非归一化模式：直接 clip 到真实速度范围 ----
        else:
            action = np.clip(action, -self.velocity_limits, self.velocity_limits)
            # 记录 last_action（这里记录的是“真实速度”）
            self.last_action = action
        # print(f'normalize_action={self.normalize_action}')
        # print(f'action shape={action.shape}')       # [1, 2]

        # print(f"action dim0 min/max: {mins[0]:.4f}, {maxs[0]:.4f}")
        # print(f"action dim1 min/max: {mins[1]:.4f}, {maxs[1]:.4f}")
        # 当前插入长度（数组化便于向量运算）
        inserted_lengths = np.array(self.device_lengths_inserted)
        # 最大长度（数组化）
        max_lengths = np.array(self.device_lengths_maximum)
        # 每步时间：用透视帧率定义 dt（例如 15Hz -> dt=1/15）
        duration = 1 / self.fluoroscopy.image_frequency
        # ---- 限制回撤不能小于 0（插入长度不能变成负）----
        # 如果 inserted + v*dt <= 0，则把推进速度 action[:,0] 置 0
        mask = np.where(inserted_lengths + action[:, 0] * duration <= 0.0)
        action[mask, 0] = 0.0
        # ---- 限制推进不能超过 max_lengths ----
        mask = np.where(inserted_lengths + action[:, 0] * duration >= max_lengths)
        action[mask, 0] = 0.0
        # 取器械 tip 位置（这里假设 dof_positions[0] 就是 tip）
        tip = self.simulation.dof_positions[0]
        # 如果开启“到末端停止”且 tip 已到血管末端，则额外限制推进
        if self.stop_device_at_tree_end and at_tree_end(tip, self.vessel_tree):
            # 当前最长的插入长度
            max_length = max(inserted_lengths)
            # 只有当最长长度 >10（单位看你们定义）才启用这套逻辑（避免一开始就触发）
            if max_length > 10:
                # dist_to_longest：每个器械距离“最长器械”的差值
                # 例如最长器械 Lmax，某器械 Li，则 dist = Lmax - Li
                dist_to_longest = -1 * inserted_lengths + max_length
                # 本步实际推进量（长度变化）= 速度 * dt
                movement = action[:, 0] * duration
                # 如果本步推进量 > 与最长器械的差值，则不允许推进（置 0）
                mask = movement > dist_to_longest
                action[mask, 0] = 0.0

        self.vessel_tree.step()
        # print(f'start simulation')
        self.simulation.step(action, duration)
        self._last_simulation_error = bool(self.simulation.simulation_error)  # 立刻缓存
        # print(f'end simulation')
        self.fluoroscopy.step()
        self.target.step()
        self.navigator.step()

    @property
    def simulation_error(self) -> bool:
        return self._last_simulation_error

    # reset：每个 episode 初始化
    def reset(
        self,
        episode_number: int = 0,        # 第几局
        seed: Optional[int] = None,     # 随机种子（可选）
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        # 给 vessel_tree 分配一个子 seed（避免各模块随机序列完全一样）
        vessel_seed = None if seed is None else self._np_random.integers(0, 2**31)
        # reset 血管树（可能随机选择某段血管/坐标系等）
        self.vessel_tree.reset(episode_number, vessel_seed)
        # 读取插入点位置与方向（用于放置器械初始状态）
        ip_pos = self.vessel_tree.insertion.position
        ip_dir = self.vessel_tree.insertion.direction

        # reset 仿真后端：把血管 mesh、插入点、器械、坐标范围等都传进去
        self.simulation.reset(
            insertion_point=ip_pos,         # 插入点位置
            insertion_direction=ip_dir,     # 插入方向
            mesh_path=self.vessel_tree.mesh_path,       # 血管 mesh 文件路径
            devices=self.devices,       # 器械列表
            coords_low=self.vessel_tree.coordinate_space.low,       # 坐标空间下界（用于归一化/裁剪）
            coords_high=self.vessel_tree.coordinate_space.high,     # 坐标空间上界
            vessel_visual_path=self.vessel_tree.visu_mesh_path,     # 血管可视化 mesh 路径
        )
        # 给 target 分配子 seed 并 reset（可能随机采样目标点）
        # ✅ 修改这里：如果传入了 target_point，用它；否则原有逻辑
        target_seed = None if seed is None else self._np_random.integers(0, 2 ** 31)
        if options and "target_point" in options:
            # 用 batch_evaluate 传入的固定目标点
            self.target.reset(
                episode_nr=episode_number,
                seed=target_seed,
                coordinates=options["target_point"]
            )
        else:
            # 原有逻辑（随机采样或默认点）
            self.target.reset(episode_number, target_seed)
        # self.target.reset(episode_number, target_seed)
        # reset 透视模块（通常会清空缓存、重置相机参数等）
        self.fluoroscopy.reset(episode_number)
        self.navigator.reset()
        # 清空 last_action（归零）
        self.last_action *= 0.0

    # 更新缓存的 inserted_lengths/rotations（但目前 property 直接读 simulation，这个函数不一定被用到）
    def _update_states(self):
        self._device_lengths_inserted = self.simulation.inserted_lengths
        self._device_rotations = self.simulation.rotations

    # 关闭资源：委托给 simulation（例如关闭 SOFA/子进程/窗口等）
    def close(self) -> None:
        self.simulation.close()

    # 重置器械：委托给 simulation（把器械回到插入点、长度/旋转归零等）
    def reset_devices(self) -> None:
        self.simulation.reset_devices()
