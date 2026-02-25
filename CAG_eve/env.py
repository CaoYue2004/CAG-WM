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
from .intervention import Intervention      
from .observation import Observation, ObsDict, ObsTuple     
from .reward import Reward      
from .terminal import Terminal      
from .truncation import Truncation, TruncationDummy     
from .info import Info, InfoDummy       
from .util import EveObject, ConfigHandler      
from .visualisation import Visualisation, VisualisationDummy        
from .start import Start, InsertionPoint        
from .pathfinder import Pathfinder, PathfinderDummy     
from .interimtarget import InterimTarget, InterimTargetDummy    
from .intervention.simulation.sofa_process_wrapper import SofaBeamAdapterProcess


ObsType = TypeVar(
    "ObsType",
    np.ndarray,
    List[np.ndarray],
    Dict[str, np.ndarray],
)
RenderFrame = TypeVar("RenderFrame")


DEFAULT_BRANCHES = []


def _pack_state(obs_dict):
    pos = obs_dict["position"].reshape(-1)
    tgt = obs_dict["target"].reshape(-1)
    rot = obs_dict["rotation"].reshape(-1)
    state = np.concatenate([pos, tgt, rot], axis=0).astype(np.float32)  # (14,)
    return {"state": state}

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
        self.start = start or InsertionPoint(intervention)
        self.visualisation = visualisation or VisualisationDummy()
        self.pathfinder = pathfinder or PathfinderDummy()
        self.interim_target = interim_target or InterimTargetDummy()

        self.episode_number = 0

    @property
    def observation_space(self) -> gym.Space:
        return self.observation.space

    @property
    def action_space(self) -> gym.Space:
        return self.intervention.action_space

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        if hasattr(self.truncation, "before_step"):
            self.truncation.before_step()
            # print(f'have before step')
        self.intervention.step(action)
        # print(f'end intervention step')
        self.pathfinder.step()
        self.interim_target.step()
        self.observation.step()
        self.reward.step()

        self.terminal.step()
        self.truncation.step()
        self.info.step()

        info_dict = deepcopy(self.info.info)
        info_dict['success'] = bool(self.intervention.target.reached)
        info_dict["truncated"] = bool(self.truncation.truncated)
        info_dict["terminated"] = bool(self.terminal.terminal)
        info_dict["simulation_error"] = bool(getattr(self.intervention, "simulation_error", False))
        info_dict["distance"] = self.pathfinder.path_length
        info_dict["position"] = self.intervention.fluoroscopy.tracking3d[0]

        return (
            deepcopy(self.observation()),
            deepcopy(self.reward.reward),
            deepcopy(self.terminal.terminal),
            deepcopy(self.truncation.truncated),
            info_dict,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,     
        options: Optional[Dict[str, Any]] = None,   
    ) -> Tuple[ObsType, Dict[str, Any]]:
        super().reset(seed=seed)
        self.intervention.reset(self.episode_number, seed, options)
        self.start.reset(self.episode_number)
        self.pathfinder.reset(self.episode_number)
        self.interim_target.reset(self.episode_number)
        self.observation.reset(self.episode_number)
        self.reward.reset(self.episode_number)
        self.terminal.reset(self.episode_number)
        self.truncation.reset(self.episode_number)
        self.info.reset(self.episode_number)
        self.visualisation.reset(self.episode_number)
        self.episode_number += 1
        return (
            deepcopy(self.observation()),
            deepcopy(self.info()),
        )

    def render(self) -> Optional[np.ndarray]:
        return self.visualisation.render()

    def close(self):
        self.intervention.close()
        self.visualisation.close()


def _load_config_maybe(cfg):
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

    target = CAG_eve.intervention.target.FixedPoint3D(fluoroscopy=fluoroscopy, threshold=getattr(cfg, "target_threshold", 5))

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


