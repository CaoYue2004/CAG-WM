from typing import List

from .reward import Reward


# ----------------------------------------------------------------------
# Combination：组合奖励
# 作用：
# - 接收多个 Reward 实例
# - reset 时依次 reset 每个子奖励，并清零总 reward
# - step 时依次 step 每个子奖励，把它们的 reward 累加成总 reward
# ----------------------------------------------------------------------
class Combination(Reward):
    # ------------------------------------------------------------------
    # 构造函数：传入一个 Reward 列表
    # ------------------------------------------------------------------
    def __init__(self, rewards: List[Reward]) -> None:
        self.rewards = rewards

    # ------------------------------------------------------------------
    # reset：重置所有子奖励，并将组合 reward 置 0
    # ------------------------------------------------------------------
    def reset(self, episode_nr: int = 0) -> None:
        for reward in self.rewards:
            reward.reset(episode_nr)
        self.reward = 0.0

    # ------------------------------------------------------------------
    # step：更新所有子奖励，并将它们的 reward 累加
    # ------------------------------------------------------------------
    def step(self) -> None:
        cum_reward = 0
        for reward in self.rewards:
            reward.step()
            cum_reward += reward.reward
        self.reward = cum_reward
