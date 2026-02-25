# eve/navigation/navigator.py
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...util import EveObject


class Navigator(EveObject, ABC):
    target_point_world: np.ndarray              
    dist_to_target_scalar: float                
    route_pts_world: np.ndarray                 
    route_pts_local: np.ndarray                 
    k_route: int                               

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def step(
        self,
    ) -> None:
        ...

    def get_reset_state(self) -> Dict[str, Any]:
        state = {
            "target_point_world": getattr(self, "target_point_world", None),
            "k_route": getattr(self, "k_route", None),
        }
        return deepcopy(state)

    def get_step_state(self) -> Dict[str, Any]:
        state = {
            "dist_to_target_scalar": getattr(self, "dist_to_target_scalar", None),
            "route_pts_world": getattr(self, "route_pts_world", None),
            "route_pts_local": getattr(self, "route_pts_local", None),
        }

        return deepcopy(state)
