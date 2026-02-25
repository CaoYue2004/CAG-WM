from .reward import Reward
from ..intervention import Intervention


# ----------------------------------------------------------------------
# InsertionLengthRelative：相对插入长度的奖励/惩罚项
# 作用：
# - 计算某个设备(device_id)与参考设备(relative_to_device_id)的插入长度差
# - 如果差值落在 [lower_clearance, upper_clearance] 的“安全区间”内 → reward=0
# - 否则按 |relative_length| * factor 给惩罚（注意这里是正值，语义更像 penalty）
# ----------------------------------------------------------------------
class InsertionLengthRelative(Reward):
    # ------------------------------------------------------------------
    # 构造函数：配置要比较的设备、惩罚系数、以及允许的安全区间
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # step：每一步更新 reward
    # ------------------------------------------------------------------
    def step(self) -> None:
        # 读取所有设备的当前插入长度
        inserted_lengths = self.intervention.device_lengths_inserted
        # 计算相对插入长度：当前设备 - 参考设备
        relative_length = (
            inserted_lengths[self.device_id]
            - inserted_lengths[self.relative_to_device_id]
        )

        # 如果 relative_length 落在 (lower_clearance, upper_clearance) 内，奖励为 0（无惩罚）
        if self.upper_clearance > relative_length > self.lower_clearance:
            self.reward = 0.0
        # 否则：按偏离程度给惩罚（绝对值越大惩罚越大）
        else:
            self.reward = abs(relative_length) * self.factor

    def reset(self, episode_nr: int = 0) -> None:
        self.reward = 0.0
