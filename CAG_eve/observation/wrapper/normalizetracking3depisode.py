from typing import Union
import numpy as np
from ..tracking3d import Tracking3D
from ..target3d import Target3D
from ..routeptslocal import RoutePtsLocal
from ...intervention import Intervention
from .normalize import Normalize, Optional

class NormalizeTracking3DEpisode(Normalize):
    def __init__(
            self,
            wrapped_obs: Union[Tracking3D, Target3D, RoutePtsLocal],
            intervention: Intervention,
            name: Optional[str] = None,
    ) -> None:
        super().__init__(wrapped_obs, name)
        self.intervention = intervention
        self._normalization_space = None

    def reset(self, episode_nr: int = 0) -> None:
        self._normalization_space = (
            self.intervention.fluoroscopy.tracking3d_space_episode
        )
        return super().reset(episode_nr)

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        low = self._normalization_space.low
        high = self._normalization_space.high
        return np.array(2 * ((obs - low) / (high - low)) - 1, dtype=np.float32)
