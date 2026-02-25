from .reward import Reward
from ..intervention import Intervention


# ----------------------------------------------------------------------
# LastAction：基于“上一动作”的奖励
# 作用：
# - 从 intervention.last_action 中取指定维度的动作值
# - 乘以缩放因子 factor 作为当前 step 的 reward
# - 常用于：鼓励/惩罚某一维动作大小（如速度、推进量、旋转角等）
# ----------------------------------------------------------------------
class LastAction(Reward):
    # ------------------------------------------------------------------
    # 构造函数
    # 参数：
    # - intervention : Intervention
    #     环境/介入对象，内部维护 last_action
    # - action_idx : int
    #     需要使用的动作维度索引
    # - factor : float
    #     奖励缩放因子（reward = last_action[action_idx] * factor）
    # ------------------------------------------------------------------
    def __init__(
        self, intervention: Intervention, action_idx: int, factor: float
    ) -> None:
        self.intervention = intervention

        self.action_idx = action_idx
        self.factor = factor

        # --------------------------------------------------------------
        # 合法性检查：
        # action_idx 必须落在 last_action 的维度范围内
        # --------------------------------------------------------------
        if self.action_idx > len(self.intervention.last_action) - 1:
            raise ValueError(
                f"Speed Index needs to map to the speed Tuple of the device. speed_idx: {self.action_idx} cannot be used. It needs to be between 0 and {len(self.intervention.speed)-1} speed values available"
            )

    def step(self) -> None:
        last_action = self.intervention.last_action
        self.reward = last_action[self.action_idx] * self.factor

    def reset(self, episode_nr: int = 0) -> None:
        self.reward = 0.0
