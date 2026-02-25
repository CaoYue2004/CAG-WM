from .reward import Reward
from ..intervention import Intervention


# ----------------------------------------------------------------------
# InsertionLengthRelativeDelta：相对插入长度“变化量”的奖励/惩罚项
# 作用：
# - 计算 device_id 相对 relative_to_device_id 的插入长度差 relative_length
# - 若 relative_length 落在安全区间 (lower_clearance, upper_clearance) 内：reward = 0
# - 否则根据 |relative_length| 相比上一时刻的“变化量”给奖励/惩罚：
#     delta = |relative_length| - |last_relative_length|
#     reward = delta * factor
#   含义：
#   - 如果偏离在变大（delta>0）→ reward 为正（更像惩罚）
#   - 如果偏离在变小（delta<0）→ reward 为负（更像奖励）
#   注意：这里 reward 的正负取决于 factor 的正负以及 delta 的符号
# ----------------------------------------------------------------------
class InsertionLengthRelativeDelta(Reward):
    def __init__(
        self,
        intervention: Intervention,
        device_id: int,
        relative_to_device_id: int,
        factor: float,
        lower_clearance: float,
        upper_clearance: float,
    ) -> None:
        self.intervention = intervention
        self.device_id = device_id
        self.relative_to_device_id = relative_to_device_id
        self.factor = factor
        self.lower_clearance = lower_clearance
        self.upper_clearance = upper_clearance
        self._last_relative_length = 0.0

    def step(self) -> None:
        inserted_lengths = self.intervention.device_lengths_inserted
        relative_length = (
            inserted_lengths[self.device_id]
            - inserted_lengths[self.relative_to_device_id]
        )

        if self.upper_clearance > relative_length > self.lower_clearance:
            self.reward = 0.0
        else:
            delta = abs(relative_length) - abs(self._last_relative_length)
            self.reward = delta * self.factor
        self._last_relative_length = relative_length

    def reset(self, episode_nr: int = 0) -> None:
        self.reward = 0.0
