# pylint: disable=unused-argument
from typing import Any, Dict, List, Optional
import gymnasium as gym

import numpy as np
from .intervention import SimulatedIntervention
from .target import Target
from .vesseltree import VesselTree
from .vesseltree.vesseltree import at_tree_end
from .fluoroscopy import SimulatedFluoroscopy
from .device import Device
from .simulation import Simulation
from .navigator import GraphNavigator


class MonoPlaneStatic(SimulatedIntervention):
    def __init__(
        self,
        vessel_tree: VesselTree,        
        devices: List[Device],          
        simulation: Simulation,         
        fluoroscopy: SimulatedFluoroscopy,      
        target: Target,         
        navigator: GraphNavigator,
        stop_device_at_tree_end: bool = True,       
        normalize_action: bool = False,         
    ) -> None:
        self.vessel_tree = vessel_tree
        self.devices = devices
        self.target = target
        self.navigator = navigator
        self.fluoroscopy = fluoroscopy
        self.stop_device_at_tree_end = stop_device_at_tree_end
        self.normlaize_action = normalize_action
        self.simulation = simulation
        self._np_random = np.random.default_rng()

        self.velocity_limits = np.array(
            [device.velocity_limit for device in self.devices]
        )
        self.last_action = np.zeros_like(self.velocity_limits)
        self._device_lengths_inserted = self.simulation.inserted_lengths
        self._device_rotations = self.simulation.rotations
        self._last_simulation_error = False

    @property
    def device_lengths_inserted(self) -> List[float]:
        return self.simulation.inserted_lengths

    @property
    def device_rotations(self) -> List[float]:
        return self.simulation.rotations

    @property
    def device_lengths_maximum(self) -> List[float]:
        return [device.length for device in self.devices]

    @property
    def device_diameters(self) -> List[float]:
        return [device.sofa_device.radius * 2 for device in self.devices]

    @property
    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self.velocity_limits.shape,
            dtype=np.float32
        )

    def step(self, action: np.ndarray) -> None:
        action = np.array(action).reshape(self.velocity_limits.shape)
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
        else:
            action = np.clip(action, -self.velocity_limits, self.velocity_limits)
            self.last_action = action
        # print(f'normalize_action={self.normalize_action}')
        # print(f'action shape={action.shape}')       # [1, 2]

        # print(f"action dim0 min/max: {mins[0]:.4f}, {maxs[0]:.4f}")
        # print(f"action dim1 min/max: {mins[1]:.4f}, {maxs[1]:.4f}")
        inserted_lengths = np.array(self.device_lengths_inserted)
        max_lengths = np.array(self.device_lengths_maximum)
        duration = 1 / self.fluoroscopy.image_frequency
        mask = np.where(inserted_lengths + action[:, 0] * duration <= 0.0)
        action[mask, 0] = 0.0
        mask = np.where(inserted_lengths + action[:, 0] * duration >= max_lengths)
        action[mask, 0] = 0.0
        tip = self.simulation.dof_positions[0]
        if self.stop_device_at_tree_end and at_tree_end(tip, self.vessel_tree):
            max_length = max(inserted_lengths)
            if max_length > 10:
                dist_to_longest = -1 * inserted_lengths + max_length
                movement = action[:, 0] * duration
                mask = movement > dist_to_longest
                action[mask, 0] = 0.0

        self.vessel_tree.step()
        # print(f'start simulation')
        self.simulation.step(action, duration)
        self._last_simulation_error = bool(self.simulation.simulation_error)  
        # print(f'end simulation')
        self.fluoroscopy.step()
        self.target.step()
        self.navigator.step()

    @property
    def simulation_error(self) -> bool:
        return self._last_simulation_error

    def reset(
        self,
        episode_number: int = 0,        
        seed: Optional[int] = None,     
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        vessel_seed = None if seed is None else self._np_random.integers(0, 2**31)
        self.vessel_tree.reset(episode_number, vessel_seed)
        ip_pos = self.vessel_tree.insertion.position
        ip_dir = self.vessel_tree.insertion.direction

        self.simulation.reset(
            insertion_point=ip_pos,         
            insertion_direction=ip_dir,     
            mesh_path=self.vessel_tree.mesh_path,       
            devices=self.devices,       
            coords_low=self.vessel_tree.coordinate_space.low,       
            coords_high=self.vessel_tree.coordinate_space.high,     
            vessel_visual_path=self.vessel_tree.visu_mesh_path,     
        )
        target_seed = None if seed is None else self._np_random.integers(0, 2 ** 31)
        if options and "target_point" in options:
            self.target.reset(
                episode_nr=episode_number,
                seed=target_seed,
                coordinates=options["target_point"]
            )
        else:
            self.target.reset(episode_number, target_seed)
        self.fluoroscopy.reset(episode_number)
        self.navigator.reset()
        self.last_action *= 0.0

    def _update_states(self):
        self._device_lengths_inserted = self.simulation.inserted_lengths
        self._device_rotations = self.simulation.rotations

    def close(self) -> None:
        self.simulation.close()

    def reset_devices(self) -> None:
        self.simulation.reset_devices()
