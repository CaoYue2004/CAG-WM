from typing import Optional
from .reward import Reward
from ..intervention import Intervention
from ..interimtarget import InterimTarget, InterimTargetDummy


class TargetReached(Reward):
    def __init__(
        self,
        intervention: Intervention,
        final_factor: float,
        interim_factor: float = 0.2,
        interim_target: Optional[InterimTarget] = None,
        final_only_after_all_interim: bool = False,
    ) -> None:
        self.intervention = intervention

        self.final_factor = final_factor
        self.interim_factor = interim_factor

        self.interim_target = interim_target or InterimTargetDummy()
        self.final_only_after_all_interim = final_only_after_all_interim

        self._interim_reward_given = False
        self._final_reward_given = False

    def step(self) -> None:
        reward = 0.0

        if (
            self.interim_target is not None
            and self.interim_target.reached
            and not self._interim_reward_given
        ):
            reward += self.interim_factor
            self._interim_reward_given = True

        final_reached = self.intervention.target.reached

        if self.final_only_after_all_interim:
            final_reached = final_reached and self._interim_reward_given

        if final_reached and not self._final_reward_given:
            reward += self.final_factor
            self._final_reward_given = True

        self.reward = reward

    def reset(self, episode_nr: int = 0) -> None:
        self.reward = 0.0
        self._interim_reward_given = False
        self._final_reward_given = False

