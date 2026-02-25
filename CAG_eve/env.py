from copy import deepcopy
from importlib import import_module
from typing import List, Tuple, Dict, Any, Optional, TypeVar, Union
import numpy as np
import gymnasium as gym
from pathlib import Path
import yaml
import json
from omegaconf import OmegaConf, DictConfig

from CAG_eve.wrappers.timeout import Timeout
import CAG_eve
from CAG_eve.visualisation.sofapygame import SofaPygame
from CAG_eve.visualisation.CAGReneder import CAGRender
from .intervention import Intervention      # 仿真本体：血管介入过程
from .observation import Observation, ObsDict, ObsTuple     # 观测模块及其组合形式
from .reward import Reward      # 奖励模块
from .terminal import Terminal      # 终止判定模块（成功/失败等）
from .truncation import Truncation, TruncationDummy     # 截断模块（超时/超步数等）及空实现
from .info import Info, InfoDummy       # info 模块（调试信息）及空实现
from .util import EveObject, ConfigHandler      # 框架基类/配置解析工具
from .visualisation import Visualisation, VisualisationDummy        # 可视化模块及空实现
from .start import Start, InsertionPoint        # 起始位置设置模块，默认插入点
from .pathfinder import Pathfinder, PathfinderDummy     # 寻路/中心线路径模块及空实现
from .interimtarget import InterimTarget, InterimTargetDummy    # 中间目标模块及空实现
from .intervention.simulation.sofa_process_wrapper import SofaBeamAdapterProcess


# ObsType：观测可能是 ndarray、ndarray list、或 dict[str, ndarray]
ObsType = TypeVar(
    "ObsType",
    np.ndarray,
    List[np.ndarray],
    Dict[str, np.ndarray],
)
# RenderFrame：渲染返回的帧类型（这里只声明，实际没用到）
RenderFrame = TypeVar("RenderFrame")


DEFAULT_BRANCHES = []


def _pack_state(obs_dict):
    pos = obs_dict["position"].reshape(-1)
    tgt = obs_dict["target"].reshape(-1)
    rot = obs_dict["rotation"].reshape(-1)
    state = np.concatenate([pos, tgt, rot], axis=0).astype(np.float32)  # (14,)
    return {"state": state}

# =========================
# 完整 RL 环境：符合 gym.Env 接口
# =========================
class Env(gym.Env, EveObject):
    def __init__(
        self,
        intervention: Intervention,
        observation: Union[Observation, ObsDict, ObsTuple],
        reward: Reward,
        terminal: Terminal,
        truncation: Optional[Truncation],
        info: Optional[Info] = None,
        start: Optional[Start] = None,
        pathfinder: Optional[Pathfinder] = None,
        interim_target: Optional[InterimTarget] = None,
        visualisation: Optional[Visualisation] = None,
    ) -> None:
        self.intervention = intervention
        self.observation = observation
        self.reward = reward
        self.terminal = terminal
        self.truncation = truncation or TruncationDummy()
        self.info = info or InfoDummy()
        # start：如果没提供则默认用插入点 InsertionPoint（依赖 intervention 提供 insertion point）
        self.start = start or InsertionPoint(intervention)
        # visualisation：如果没提供就用空实现（render 返回 None 或空画面）
        self.visualisation = visualisation or VisualisationDummy()
        self.pathfinder = pathfinder or PathfinderDummy()
        self.interim_target = interim_target or InterimTargetDummy()

        # 记录 episode 编号：每 reset 一次 +1，供各模块决定随机/目标等
        self.episode_number = 0

    # gym.Env 要求提供 observation_space 属性
    @property
    def observation_space(self) -> gym.Space:
        # observation 模块自己暴露 space
        return self.observation.space

    # gym.Env 要求提供 action_space 属性
    @property
    def action_space(self) -> gym.Space:
        # action space 来自 intervention（因为 action 直接作用到仿真）
        return self.intervention.action_space

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        if hasattr(self.truncation, "before_step"):
            self.truncation.before_step()
            # print(f'have before step')
        # 1) 先推进仿真：动作真正起作用的地方
        self.intervention.step(action)
        # print(f'end intervention step')
        # 2) 根据新状态更新路径（例如中心线到目标的最短路径）
        self.pathfinder.step()
        # 3) 更新中间目标（例如分段导航目标）
        self.interim_target.step()
        # 4) 更新观测（把当前状态转成 agent 能看到的 obs）
        self.observation.step()
        # 5) 更新奖励（从状态/路径/目标等计算 reward）
        self.reward.step()

        # ... step() 里
        self.terminal.step()
        self.truncation.step()
        # 8) 更新 info（记录调试信息/额外指标）
        self.info.step()

        info_dict = deepcopy(self.info.info)
        info_dict['success'] = bool(self.intervention.target.reached)
        info_dict["truncated"] = bool(self.truncation.truncated)
        info_dict["terminated"] = bool(self.terminal.terminal)
        # 如果 intervention 有错误标志：
        info_dict["simulation_error"] = bool(getattr(self.intervention, "simulation_error", False))
        info_dict["distance"] = self.pathfinder.path_length
        info_dict["position"] = self.intervention.fluoroscopy.tracking3d[0]

        # 返回：都 deepcopy，防止外部拿到引用后修改内部对象
        return (
            deepcopy(self.observation()),
            deepcopy(self.reward.reward),
            deepcopy(self.terminal.terminal),
            deepcopy(self.truncation.truncated),
            info_dict,
        )

    # gym.Env 的 reset：返回 (obs, info)
    def reset(
        self,
        *,
        seed: Optional[int] = None,     # gymnasium 规范：reset 可传 seed
        options: Optional[Dict[str, Any]] = None,   # reset 的额外参数
    ) -> Tuple[ObsType, Dict[str, Any]]:
        # 让 gymnasium 处理 seed（会设置 self.np_random 等）
        super().reset(seed=seed)
        # intervention reset：这里通常会重置仿真、放置器械、采样目标等
        # 注意它拿到 episode_number/seed/options
        self.intervention.reset(self.episode_number, seed, options)
        # start reset：设置起点（默认插入点），依赖 intervention 已 reset 好
        self.start.reset(self.episode_number)
        # pathfinder reset：准备路径搜索的初始状态
        self.pathfinder.reset(self.episode_number)
        # interim target reset：准备中间目标
        self.interim_target.reset(self.episode_number)
        # observation reset：准备观测缓存/归一化等
        self.observation.reset(self.episode_number)
        # reward reset：清空累计量/初始化 reward
        self.reward.reset(self.episode_number)
        # terminal reset：清空终止标志
        self.terminal.reset(self.episode_number)
        # truncation reset：清空终止标志
        self.truncation.reset(self.episode_number)
        # info reset：清空 info
        self.info.reset(self.episode_number)
        # visualisation reset：清空可视化窗口/对象
        self.visualisation.reset(self.episode_number)
        # episode 编号 +1，下一次 reset 就是新的一局
        self.episode_number += 1
        return (
            deepcopy(self.observation()),
            deepcopy(self.info()),
        )

    # gym.Env 的 render：返回渲染帧（或 None）
    def render(self) -> Optional[np.ndarray]:
        return self.visualisation.render()

    # gym.Env 的 close：释放资源
    def close(self):
        # 关闭仿真资源（例如 SOFA、文件句柄等）
        self.intervention.close()
        # 关闭可视化资源（窗口/线程等）
        self.visualisation.close()


def _load_config_maybe(cfg):
    """
    支持两种来源：
    1) cfg.env_config: 已经是 dict
    2) cfg.env_config_path: 指向 yaml/json 文件
    """
    if hasattr(cfg, "env_config") and cfg.env_config is not None:
        return deepcopy(cfg.env_config)

    if hasattr(cfg, "env_config_path") and cfg.env_config_path is not None:
        p = Path(cfg.env_config_path)
        if not p.exists():
            raise FileNotFoundError(f"env_config_path not found: {p}")
        if p.suffix in [".yml", ".yaml"]:
            if yaml is None:
                raise ImportError("PyYAML not installed but a .yaml config was provided.")
            return yaml.safe_load(p.read_text(encoding="utf-8"))
        if p.suffix == ".json":
            return json.loads(p.read_text(encoding="utf-8"))
        raise ValueError(f"Unsupported config file suffix: {p.suffix}")

    raise ValueError("Need cfg.env_config (dict) or cfg.env_config_path (yaml/json).")


def make_cag_env(cfg):
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    # 1) vessel tree（必须参数 + 默认参数）
    vessel_tree = CAG_eve.intervention.vesseltree.CoronaryArtery(
        model_folder=cfg.model_folder,
        insertion_vessel_name=cfg.insertion_vessel_name,
        insertion_point_idx=cfg.insertion_point_idx,
        insertion_direction_idx_diff=getattr(cfg, "insertion_direction_idx_diff", 1),
        approx_branch_radii=getattr(cfg, "approx_branch_radii", 5.0),
        check_if_points_in_mesh=getattr(cfg, "check_if_points_in_mesh", False),
    )

    # 2) device/sim/fluoro
    device = CAG_eve.intervention.device.JShaped()
    simulation = SofaBeamAdapterProcess(friction=0.1, dt_simulation=0.002, step_timeout_s=10.0)

    fluoroscopy = CAG_eve.intervention.fluoroscopy.TrackingOnly(
        simulation=simulation,
        vessel_tree=vessel_tree,
        image_frequency=getattr(cfg, "image_frequency", 7.5),
        image_rot_zx=getattr(cfg, "image_rot_zx", [20, 5]),
    )

    # 3) target（branches 默认内置，可覆盖）
    target = CAG_eve.intervention.target.CenterlineRandom(
        vessel_tree=vessel_tree,
        fluoroscopy=fluoroscopy,
        threshold=getattr(cfg, "target_threshold", 5),
        branches=getattr(cfg, "branches", DEFAULT_BRANCHES),
    )
    # print(f'branch={vessel_tree.branches}')

    navigator = CAG_eve.intervention.navigator.GraphNavigator(vessel_tree=vessel_tree, target=target, simulation=simulation, round_ndigits=1)

    # 4) intervention
    intervention = CAG_eve.intervention.MonoPlaneStatic(
        vessel_tree=vessel_tree,
        devices=[device],
        simulation=simulation,
        fluoroscopy=fluoroscopy,
        target=target,
        navigator=navigator,
        normalize_action=True,
    )

    # 5) helpers
    start = CAG_eve.start.InsertionPoint(intervention=intervention)
    pathfinder = CAG_eve.pathfinder.BruteForceBFS(intervention=intervention)

    # 10) interimtarget
    interimtarget = CAG_eve.interimtarget.Even(pathfinder=pathfinder, intervention=intervention,
                                               resolution=getattr(cfg, "resolution", 20),
                                               threshold=getattr(cfg, "target_threshold", 5))

    # 6) observation
    pos = CAG_eve.observation.Tracking3D(intervention=intervention, n_points=getattr(cfg, "n_points", 5))
    pos = CAG_eve.observation.wrapper.NormalizeTracking3DEpisode(pos, intervention)

    tgt = CAG_eve.observation.Target3D(intervention=intervention)
    tgt = CAG_eve.observation.wrapper.NormalizeTracking3DEpisode(tgt, intervention)

    rot = CAG_eve.observation.Rotations(intervention=intervention)

    route = CAG_eve.observation.RoutePtsLocal(intervention=intervention)
    route = CAG_eve.observation.wrapper.NormalizeTracking3DEpisode(route, intervention)

    state = CAG_eve.observation.ObsDict({"position": pos, "target": tgt, "rotation": rot, "route": route})

    # 7) reward
    target_reward = CAG_eve.reward.TargetReached(intervention=intervention, final_factor=getattr(cfg, "r_target", 1.0), interim_target=interimtarget)
    path_delta = CAG_eve.reward.PathLengthDelta(pathfinder=pathfinder, factor=getattr(cfg, "r_path", 0.1))
    error_reward = CAG_eve.reward.ComputeFailed(intervention=intervention, factor=getattr(cfg, "r_error", 1.0))
    stuck_penalty = CAG_eve.reward.StuckPenalty(intervention=intervention)
    no_progress_penalty = CAG_eve.reward.NoProgressPenalty(pathfinder=pathfinder)
    reward = CAG_eve.reward.Combination([target_reward, path_delta, error_reward, stuck_penalty, no_progress_penalty])

    # 8) terminal/truncation
    targetterminal = CAG_eve.terminal.TargetReached(intervention=intervention)
    computeterminal = CAG_eve.terminal.ComputeFailed(intervention=intervention)
    terminal = CAG_eve.terminal.Combination([targetterminal, computeterminal])
    steptruncation = CAG_eve.truncation.MaxSteps(getattr(cfg, "max_steps", 300))
    timetruncation = CAG_eve.truncation.MaxStepTime(getattr(cfg, "max_step_time", 10))
    truncation = CAG_eve.truncation.Combination([steptruncation, timetruncation])

    # 9) visualisation（可选开关）
    # 用子进程渲染器输出 frame 给 wandb
    visualisation = SofaPygame(intervention=intervention, interim_target=interimtarget) if getattr(cfg, "vis", True) else CAG_eve.visualisation.VisualisationDummy()


    # 11) env
    env = CAG_eve.Env(
        intervention=intervention,
        observation=state,
        reward=reward,
        terminal=terminal,
        truncation=truncation,
        visualisation=visualisation,
        start=start,
        pathfinder=pathfinder,
        interim_target=interimtarget,
    )
    return env


def make_eval_env(cfg):
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    # 1) vessel tree（必须参数 + 默认参数）
    vessel_tree = CAG_eve.intervention.vesseltree.CoronaryArtery(
        model_folder=cfg.model_folder,
        insertion_vessel_name=cfg.insertion_vessel_name,
        insertion_point_idx=cfg.insertion_point_idx,
        insertion_direction_idx_diff=getattr(cfg, "insertion_direction_idx_diff", 1),
        approx_branch_radii=getattr(cfg, "approx_branch_radii", 5.0),
        check_if_points_in_mesh=getattr(cfg, "check_if_points_in_mesh", False),
    )

    # 2) device/sim/fluoro
    device = CAG_eve.intervention.device.JShaped()
    simulation = CAG_eve.intervention.simulation.SofaBeamAdapter()

    fluoroscopy = CAG_eve.intervention.fluoroscopy.TrackingOnly(
        simulation=simulation,
        vessel_tree=vessel_tree,
        image_frequency=getattr(cfg, "image_frequency", 7.5),
        image_rot_zx=getattr(cfg, "image_rot_zx", [20, 5]),
    )

    # 3) target（branches 默认内置，可覆盖）
    target = CAG_eve.intervention.target.CenterlineRandom(
        vessel_tree=vessel_tree,
        fluoroscopy=fluoroscopy,
        threshold=getattr(cfg, "target_threshold", 5),
        branches=getattr(cfg, "branches", DEFAULT_BRANCHES),
    )

    navigator = CAG_eve.intervention.navigator.GraphNavigator(vessel_tree=vessel_tree, target=target,
                                                              simulation=simulation, round_ndigits=1)

    # 4) intervention
    intervention = CAG_eve.intervention.MonoPlaneStatic(
        vessel_tree=vessel_tree,
        devices=[device],
        simulation=simulation,
        fluoroscopy=fluoroscopy,
        target=target,
        navigator=navigator,
        normalize_action=True,
    )

    # 5) helpers
    start = CAG_eve.start.InsertionPoint(intervention=intervention)
    pathfinder = CAG_eve.pathfinder.BruteForceBFS(intervention=intervention)

    # 10) interimtarget
    interimtarget = CAG_eve.interimtarget.Even(pathfinder=pathfinder, intervention=intervention,
                                               resolution=getattr(cfg, "resolution", 20),
                                               threshold=getattr(cfg, "target_threshold", 5))

    # 6) observation
    pos = CAG_eve.observation.Tracking3D(intervention=intervention, n_points=getattr(cfg, "n_points", 5))
    pos = CAG_eve.observation.wrapper.NormalizeTracking3DEpisode(pos, intervention)

    tgt = CAG_eve.observation.Target3D(intervention=intervention)
    tgt = CAG_eve.observation.wrapper.NormalizeTracking3DEpisode(tgt, intervention)

    rot = CAG_eve.observation.Rotations(intervention=intervention)

    route = CAG_eve.observation.RoutePtsLocal(intervention=intervention)
    route = CAG_eve.observation.wrapper.NormalizeTracking3DEpisode(route, intervention)

    state = CAG_eve.observation.ObsDict({"position": pos, "target": tgt, "rotation": rot, "route": route})

    # 7) reward
    target_reward = CAG_eve.reward.TargetReached(intervention=intervention, final_factor=getattr(cfg, "r_target", 1.0),
                                                 interim_target=interimtarget)
    path_delta = CAG_eve.reward.PathLengthDelta(pathfinder=pathfinder, factor=getattr(cfg, "r_path", 0.1))
    error_reward = CAG_eve.reward.ComputeFailed(intervention=intervention, factor=getattr(cfg, "r_error", 1.0))
    stuck_penalty = CAG_eve.reward.StuckPenalty(intervention=intervention)
    no_progress_penalty = CAG_eve.reward.NoProgressPenalty(pathfinder=pathfinder)
    reward = CAG_eve.reward.Combination([target_reward, path_delta, error_reward, stuck_penalty, no_progress_penalty])

    # 8) terminal/truncation
    targetterminal = CAG_eve.terminal.TargetReached(intervention=intervention)
    computeterminal = CAG_eve.terminal.ComputeFailed(intervention=intervention)
    terminal = CAG_eve.terminal.Combination([targetterminal, computeterminal])
    steptruncation = CAG_eve.truncation.MaxSteps(getattr(cfg, "max_steps", 300))
    simtruncation = CAG_eve.truncation.SimError(intervention=intervention)
    timetruncation = CAG_eve.truncation.MaxStepTime(getattr(cfg, "max_step_time", 10))
    truncation = CAG_eve.truncation.Combination([steptruncation, simtruncation, timetruncation])

    # 9) visualisation（可选开关）
    # 用子进程渲染器输出 frame 给 wandb
    visualisation = CAGRender(
        network_pkl=cfg.network_pkl,
        intervention=intervention,
        ppa=cfg.ppa,
        psa=cfg.psa,
        reload_modules=True,
        seed=cfg.seed,
        angles_csv=cfg.csv
    )

    # visualisation = SofaPygame(intervention=intervention, interim_target=interimtarget)

    # 11) env
    env = CAG_eve.Env(
        intervention=intervention,
        observation=state,
        reward=reward,
        terminal=terminal,
        truncation=truncation,
        visualisation=visualisation,
        start=start,
        pathfinder=pathfinder,
        interim_target=interimtarget,
    )
    return env


def make_eval_batch_env(cfg):
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    # 1) vessel tree（必须参数 + 默认参数）
    vessel_tree = CAG_eve.intervention.vesseltree.CoronaryArtery(
        model_folder=cfg.model_folder,
        insertion_vessel_name=cfg.insertion_vessel_name,
        insertion_point_idx=cfg.insertion_point_idx,
        insertion_direction_idx_diff=getattr(cfg, "insertion_direction_idx_diff", 1),
        approx_branch_radii=getattr(cfg, "approx_branch_radii", 5.0),
        check_if_points_in_mesh=getattr(cfg, "check_if_points_in_mesh", False),
    )

    # 2) device/sim/fluoro
    device = CAG_eve.intervention.device.JShaped()
    simulation = SofaBeamAdapterProcess(friction=0.1, dt_simulation=0.002, step_timeout_s=10.0)

    fluoroscopy = CAG_eve.intervention.fluoroscopy.TrackingOnly(
        simulation=simulation,
        vessel_tree=vessel_tree,
        image_frequency=getattr(cfg, "image_frequency", 7.5),
        image_rot_zx=getattr(cfg, "image_rot_zx", [20, 5]),
    )

    # 3) target（branches 默认内置，可覆盖）
    target = CAG_eve.intervention.target.FixedPoint3D(fluoroscopy=fluoroscopy, threshold=getattr(cfg, "target_threshold", 5))

    '''target = CAG_eve.intervention.target.CenterlineRandom(
        vessel_tree=vessel_tree,
        fluoroscopy=fluoroscopy,
        threshold=getattr(cfg, "target_threshold", 5),
        branches=getattr(cfg, "branches", DEFAULT_BRANCHES),
    )'''

    navigator = CAG_eve.intervention.navigator.GraphNavigator(vessel_tree=vessel_tree, target=target,
                                                              simulation=simulation, round_ndigits=1)

    # 4) intervention
    intervention = CAG_eve.intervention.MonoPlaneStatic(
        vessel_tree=vessel_tree,
        devices=[device],
        simulation=simulation,
        fluoroscopy=fluoroscopy,
        target=target,
        navigator=navigator,
        normalize_action=True,
    )

    # 5) helpers
    start = CAG_eve.start.InsertionPoint(intervention=intervention)
    pathfinder = CAG_eve.pathfinder.BruteForceBFS(intervention=intervention)

    # 10) interimtarget
    interimtarget = CAG_eve.interimtarget.Even(pathfinder=pathfinder, intervention=intervention,
                                               resolution=getattr(cfg, "resolution", 20),
                                               threshold=getattr(cfg, "target_threshold", 5))

    # 6) observation
    pos = CAG_eve.observation.Tracking3D(intervention=intervention, n_points=getattr(cfg, "n_points", 5))
    pos = CAG_eve.observation.wrapper.NormalizeTracking3DEpisode(pos, intervention)

    tgt = CAG_eve.observation.Target3D(intervention=intervention)
    tgt = CAG_eve.observation.wrapper.NormalizeTracking3DEpisode(tgt, intervention)

    rot = CAG_eve.observation.Rotations(intervention=intervention)

    route = CAG_eve.observation.RoutePtsLocal(intervention=intervention)
    route = CAG_eve.observation.wrapper.NormalizeTracking3DEpisode(route, intervention)

    state = CAG_eve.observation.ObsDict({"position": pos, "target": tgt, "rotation": rot, "route": route})

    # 7) reward
    target_reward = CAG_eve.reward.TargetReached(intervention=intervention, final_factor=getattr(cfg, "r_target", 1.0),
                                                 interim_target=interimtarget)
    path_delta = CAG_eve.reward.PathLengthDelta(pathfinder=pathfinder, factor=getattr(cfg, "r_path", 0.1))
    error_reward = CAG_eve.reward.ComputeFailed(intervention=intervention, factor=getattr(cfg, "r_error", 1.0))
    stuck_penalty = CAG_eve.reward.StuckPenalty(intervention=intervention)
    no_progress_penalty = CAG_eve.reward.NoProgressPenalty(pathfinder=pathfinder)
    reward = CAG_eve.reward.Combination([target_reward, path_delta, error_reward, stuck_penalty, no_progress_penalty])

    # 8) terminal/truncation
    targetterminal = CAG_eve.terminal.TargetReached(intervention=intervention)
    computeterminal = CAG_eve.terminal.ComputeFailed(intervention=intervention)
    terminal = CAG_eve.terminal.Combination([targetterminal, computeterminal])
    steptruncation = CAG_eve.truncation.MaxSteps(getattr(cfg, "max_steps", 300))
    simtruncation = CAG_eve.truncation.SimError(intervention=intervention)
    timetruncation = CAG_eve.truncation.MaxStepTime(getattr(cfg, "max_step_time", 10))
    truncation = CAG_eve.truncation.Combination([steptruncation, simtruncation, timetruncation])

    visualisation = SofaPygame(intervention=intervention, interim_target=interimtarget) if getattr(cfg, "vis", True) else CAG_eve.visualisation.VisualisationDummy()

    # 11) env
    env = CAG_eve.Env(
        intervention=intervention,
        observation=state,
        reward=reward,
        terminal=terminal,
        truncation=truncation,
        visualisation=visualisation,
        start=start,
        pathfinder=pathfinder,
        interim_target=interimtarget,
    )
    return env

