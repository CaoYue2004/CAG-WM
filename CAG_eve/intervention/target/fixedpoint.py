
from typing import Optional
import numpy as np

from .target import Target
from ..fluoroscopy import Fluoroscopy
from ...util.coordtransform import vessel_cs_to_tracking3d, tracking3d_to_2d


class FixedPoint3D(Target):

    def __init__(
            self,
            fluoroscopy: Fluoroscopy,
            threshold: float,
            default_point: Optional[np.ndarray] = None,
    ) -> None:
        self.fluoroscopy = fluoroscopy
        self.threshold = threshold

        if default_point is None:
            self.default_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            self.default_point = np.asarray(default_point, dtype=np.float32).copy()

        target_vessel_cs = self.default_point.copy()
        self.coordinates3d = vessel_cs_to_tracking3d(
            target_vessel_cs,
            self.fluoroscopy.image_rot_zx,
            self.fluoroscopy.image_center,
            self.fluoroscopy.field_of_view,
        )
        self.reached = False

    @property
    def coordinates2d(self) -> np.ndarray:
        return tracking3d_to_2d(self.coordinates3d)

    def reset(
            self,
            episode_nr: int = 0,
            seed: Optional[int] = None,
            coordinates: Optional[np.ndarray] = None,
    ) -> None:
        if coordinates is not None:
            target_vessel_cs = np.asarray(coordinates, dtype=np.float32).copy()
            self.coordinates3d = vessel_cs_to_tracking3d(
                target_vessel_cs,
                self.fluoroscopy.image_rot_zx,
                self.fluoroscopy.image_center,
                self.fluoroscopy.field_of_view,
            )
        else:
            target_vessel_cs = self.default_point.copy()
            self.coordinates3d = vessel_cs_to_tracking3d(
                target_vessel_cs,
                self.fluoroscopy.image_rot_zx,
                self.fluoroscopy.image_center,
                self.fluoroscopy.field_of_view,
            )

        self.reached = False

    def set_target(self, coordinates: np.ndarray) -> None:
        self.reset(coordinates=coordinates)


