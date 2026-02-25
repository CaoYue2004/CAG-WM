from copy import deepcopy
import numpy as np
import PIL.Image

from .observation import Observation, gym

from ..intervention import Intervention


# ----------------------------------------------------------------------
# Image：一种 Observation
# 作用：从 fluoroscopy 中取出当前帧图像，转换为 numpy(float32) 作为观测
# ----------------------------------------------------------------------
class Image(Observation):
    # ------------------------------------------------------------------
    # 构造：绑定 intervention，并设置观测名称
    # ------------------------------------------------------------------
    def __init__(self, intervention: Intervention, name: str = "image") -> None:
        self.name = name
        self.intervention = intervention
        self.image: PIL.Image.Image = None
        self.obs = None

    # ------------------------------------------------------------------
    # space：该观测对应的 gym 空间
    # 直接复用 fluoroscopy 定义的 image_space（Box）
    # ------------------------------------------------------------------
    @property
    def space(self) -> gym.spaces.Box:
        return self.intervention.fluoroscopy.image_space

    # ------------------------------------------------------------------
    # step：更新观测
    # - 深拷贝当前 fluoroscopy.image
    # - 转成 float32 numpy array 存入 self.obs
    # ------------------------------------------------------------------
    def step(self) -> None:
        self.image = deepcopy(self.intervention.fluoroscopy.image)
        self.obs = np.array(self.image, dtype=np.float32)

    # ------------------------------------------------------------------
    # reset：重置时直接调用 step，保证 obs 与当前帧一致
    # ------------------------------------------------------------------
    def reset(self, episode_nr: int = 0) -> None:
        self.step()
