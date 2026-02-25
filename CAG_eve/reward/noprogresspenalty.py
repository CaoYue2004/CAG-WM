from .reward import Reward
from ..pathfinder import Pathfinder


class NoProgressPenalty(Reward):
    """
    无进度惩罚：
    - 进度指标：pathfinder.path_length (越小越接近目标)
    - 若本步进度 < eps，则给惩罚 -lambda_stall
    - 连续 stall_steps 达到 K 后，可选择递增惩罚
    """
    def __init__(
        self,
        pathfinder: Pathfinder,
        eps: float = 1.0,
        lambda_stall: float = 0.01,
        k_steps: int = 10,
        escalate: bool = True,
        escalate_power: float = 1.0,
        max_penalty: float = 0.5,
    ) -> None:
        self.pathfinder = pathfinder
        self.eps = float(eps)
        self.lambda_stall = float(lambda_stall)
        self.k_steps = int(k_steps)
        self.escalate = bool(escalate)
        self.escalate_power = float(escalate_power)
        self.max_penalty = float(max_penalty)

        self._last_d = None
        self._stall_cnt = 0

    def reset(self, episode_nr: int = 0) -> None:
        self.reward = 0.0
        self._last_d = float(self.pathfinder.path_length)
        self._stall_cnt = 0

    def step(self) -> None:
        d = float(self.pathfinder.path_length)  # d_path: 剩余路径（越小越接近目标）
        improve = self._last_d - d  # >0 表示变近了

        if improve >= self.eps:
            # 有足够进度：不罚，清零 stall 计数
            self._stall_cnt = 0
            self.reward = 0.0
        else:
            # 无足够进度：惩罚 + 计数递增
            self._stall_cnt += 1

            penalty = self.lambda_stall
            if self.escalate and self._stall_cnt >= self.k_steps:
                extra = self._stall_cnt - self.k_steps + 1
                penalty = self.lambda_stall * (extra ** self.escalate_power)

            penalty = min(penalty, self.max_penalty)
            self.reward = -penalty

        self._last_d = d

