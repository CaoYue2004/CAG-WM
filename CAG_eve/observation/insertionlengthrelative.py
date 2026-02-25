import numpy as np

from ..intervention import Intervention
from .observation import Observation, gym


# ----------------------------------------------------------------------
# InsertionLengthRelative：相对插入长度观测
# 作用：
# - 观测某个器械（device_idx）的插入长度
# - 相对于另一个器械（relative_to_device_idx）的插入长度
# 即： inserted_lengths[device_idx] - inserted_lengths[relative_to_device_idx]
# ----------------------------------------------------------------------
class InsertionLengthRelative(Observation):
    # ------------------------------------------------------------------
    # 构造函数
    # - intervention：环境对象（提供 inserted_lengths / maximum 等）
    # - device_idx：要观测的器械索引
    # - relative_to_device_idx：作为参考的器械索引
    # - name：观测名称；若不传则自动生成
    # ------------------------------------------------------------------
    def __init__(
        self,
        intervention: Intervention,
        device_idx: int,
        relative_to_device_idx: int,
        name: str = None,
    ) -> None:
        name = (
            name or f"Device_length_{device_idx}_relative_to_{relative_to_device_idx}"
        )
        super().__init__(name)
        self.intervention = intervention
        self.device_idx = device_idx
        self.relative_to_device_idx = relative_to_device_idx

    # ------------------------------------------------------------------
    # space：该观测对应的 gym 空间（Box）
    # 上界：device_idx 的最大插入长度
    # 下界：-（参考设备的最大插入长度）
    # 解释：因为 relative_length 可能为负（当前设备插得比参考设备少）
    # ------------------------------------------------------------------
    @property
    def space(self) -> gym.spaces.Box:
        high = self.intervention.device_lengths_maximum[self.device_idx]
        high = np.array(high, dtype=np.float32)
        low = -self.intervention.device_lengths_maximum[self.relative_to_device_idx]
        low = np.array(low, dtype=np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    # ------------------------------------------------------------------
    # step：更新观测值
    # relative_length = inserted_lengths[device_idx] - inserted_lengths[relative_to_device_idx]
    # ------------------------------------------------------------------
    def step(self) -> None:
        inserted_lengths = self.intervention.device_lengths_inserted
        relative_length = (
            inserted_lengths[self.device_idx]
            - inserted_lengths[self.relative_to_device_idx]
        )
        self.obs = np.array(relative_length, dtype=np.float32)

    # ------------------------------------------------------------------
    # reset：重置时直接调用 step，保证 obs 与当前环境一致
    # ------------------------------------------------------------------
    def reset(self, episode_nr: int = 0) -> None:
        self.step()
