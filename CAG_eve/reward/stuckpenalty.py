from .reward import Reward
from ..intervention import Intervention
import numpy as np

class StuckPenalty(Reward):
    """
    Penalize being stuck:
      if |Δinserted| < eps_insert AND |Δtip| < eps_tip for N consecutive steps:
          reward -= penalty  (one-shot)
    """
    def __init__(
        self,
        intervention: Intervention,
        device_id: int = 0,
        eps_insert: float = 0.2,   # inserted length threshold (unit same as env, e.g., mm)
        eps_tip: float = 1.0,      # tip movement threshold (same unit as position, e.g., mm)
        window: int = 20,          # N steps
        penalty: float = 1.0,      # p
        cooldown: int = 10,        # after firing once, wait a few steps before detecting again
    ) -> None:
        self.intervention = intervention
        self.device_id = device_id

        self.eps_insert = eps_insert
        self.eps_tip = eps_tip
        self.window = window
        self.penalty = penalty
        self.cooldown = cooldown

        self._last_inserted = None
        self._last_tip = None
        self._stuck_count = 0
        self._cooldown_left = 0

    def _get_inserted(self) -> float:
        # You likely have: intervention.device_lengths_inserted -> List[float]
        return float(self.intervention.simulation.inserted_lengths[self.device_id])

    def _get_tip(self) -> np.ndarray:
        # Choose the correct one in your env:
        # Common options: intervention.devices[device_id].tip_position, intervention.tip_position, etc.
        # Here assume: intervention.device_tip_positions -> List[np.ndarray shape(3,)]
        tip = self.intervention.fluoroscopy.tracking3d[0]
        return np.asarray(tip, dtype=np.float32)

    def step(self) -> None:
        self.reward = 0.0

        inserted = self._get_inserted()
        tip = self._get_tip()

        # init
        if self._last_inserted is None or self._last_tip is None:
            self._last_inserted = inserted
            self._last_tip = tip
            self._stuck_count = 0
            return

        # cooldown after a penalty triggers
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            self._last_inserted = inserted
            self._last_tip = tip
            return

        d_insert = abs(inserted - self._last_inserted)
        d_tip = float(np.linalg.norm(tip - self._last_tip))

        if (d_insert < self.eps_insert) and (d_tip < self.eps_tip):
            self._stuck_count += 1
        else:
            self._stuck_count = 0

        if self._stuck_count >= self.window:
            self.reward = -float(self.penalty)
            self._stuck_count = 0
            self._cooldown_left = self.cooldown

        self._last_inserted = inserted
        self._last_tip = tip

    def reset(self, episode_nr: int = 0) -> None:
        self.reward = 0.0
        self._last_inserted = None
        self._last_tip = None
        self._stuck_count = 0
        self._cooldown_left = 0
