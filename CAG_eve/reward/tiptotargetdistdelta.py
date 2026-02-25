from typing import Optional
import numpy as np

from .reward import Reward
from ..intervention import Intervention
from ..interimtarget import InterimTarget


# 定义奖励类 TipToTargetDistDelta：基于“针尖到目标距离的变化量”给奖励
class TipToTargetDistDelta(Reward):
    def __init__(
        self,
        factor: float,
        intervention: Intervention,
        interim_target: Optional[InterimTarget],
    ) -> None:
        self.factor = factor
        self.intervention = intervention
        self.interim_target = interim_target
        self._last_dist = None
        self._last_target = None

    # step：每个环境 step 调用一次，计算距离变化并给奖励
    def step(self) -> None:
        # 取出导丝/器械尖端 tip 的三维坐标（tracking3d 的第 0 个点）
        tip = self.intervention.fluoroscopy.tracking3d[0]
        # 如果中间目标存在坐标，则使用中间目标作为当前 target
        if self.interim_target.coordinates3d is not None:
            target = self.interim_target.coordinates3d
        # 否则使用最终目标坐标
        else:
            target = self.intervention.target.coordinates3d

        dist = np.linalg.norm(tip - target)

        dist_delta = dist - self._last_dist
        self._last_dist = dist
        if np.all(target == self._last_target):
            self.reward = -dist_delta * self.factor
        else:
            self.reward = 0.0
            self._last_target = target

    def reset(self, episode_nr: int = 0) -> None:
        self.reward = 0.0
        tip = self.intervention.fluoroscopy.tracking3d[0]
        target = self.intervention.target.coordinates3d
        dist = np.linalg.norm(tip - target)
        self._last_dist = dist
        self._last_target = target
