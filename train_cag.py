import os
os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "egl")
os.environ["LAZY_LEGACY_OP"] = "0"
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["TORCH_LOGS"] = "+recompiles"

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import gymnasium as gym
import hydra
from omegaconf import OmegaConf
from termcolor import colored

# 你工程里的这些
from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from tdmpc2 import TDMPC2
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger

# 你的 CAG_eve 环境构建（你已经写了 make_env(cfg) 的那份）
# 假设你把它放在 CAG_eve/env.py 或 CAG_eve/envs/cag_env.py 里，按实际路径改：
from CAG_eve.env import make_cag_env


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


# ---------------------------
# 关键：把 gymnasium 5元组 Env 适配成你们 Trainer 期待的旧接口
# ---------------------------
class LegacyStepAPIWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, log_dir: str = "position_logs", flush_every: int = 1000):
        super().__init__(env)
        self._last_info: Dict[str, Any] = {}

        # ---- position 记录 ----
        self._log_dir = log_dir
        self._flush_every = flush_every  # 每N步写一次磁盘
        self._position_log: list = []  # 内存缓冲
        self._total_steps: int = 0  # 全局步数计数
        self._ep_idx: int = 0
        self._step_idx: int = 0

        # 创建输出目录
        os.makedirs(log_dir, exist_ok=True)
        self._csv_path = os.path.join(log_dir, "position_log.csv")

        # 如果文件不存在，写入header
        if not os.path.exists(self._csv_path):
            import csv
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["episode", "step", "x", "y", "z", "success", "terminated"]
                )
                writer.writeheader()
        # -----------------------

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        self._last_info = dict(info) if isinstance(info, dict) else {}
        self._ep_idx += 1
        self._step_idx = 0
        return obs

    def step(self, action):
        out = self.env.step(action)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
            info = dict(info) if isinstance(info, dict) else {}
            info.setdefault("terminated", bool(terminated))
            info.setdefault("truncated", bool(truncated))
            info.setdefault("done_for_learning", bool(terminated))
        elif isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            info = dict(info) if isinstance(info, dict) else {}
            info.setdefault("terminated", bool(done))
            info.setdefault("truncated", False)
            info.setdefault("done_for_learning", bool(done))
        else:
            raise RuntimeError(f"Unexpected env.step return: {type(out)}")

        info.setdefault("success", bool(info.get("is_success", False)))
        info.setdefault("is_success", bool(info.get("success", False)))

        # ---- 记录 position ----
        self._step_idx += 1
        self._total_steps += 1
        pos = info.get("position", None)
        if pos is not None:
            pos = np.asarray(pos, dtype=np.float32).flatten()
            self._position_log.append({
                "episode": self._ep_idx,
                "step": self._step_idx,
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]) if len(pos) > 2 else 0.0,
                "success": int(info.get("success", False)),
                "terminated": int(info.get("terminated", False)),
            })

            # episode结束 或 每1000步 → flush
            if done or self._total_steps % self._flush_every == 0:
                self._flush()
        # ----------------------

        self._last_info = info
        return obs, float(reward), bool(done), info

    def _flush(self):
        """把内存缓冲追加写入 CSV，然后清空缓冲"""
        if not self._position_log:
            return
        import csv
        with open(self._csv_path, "a", newline="") as f:  # 'a' = 追加模式
            writer = csv.DictWriter(
                f, fieldnames=["episode", "step", "x", "y", "z", "success", "terminated"]
            )
            writer.writerows(self._position_log)

        n = len(self._position_log)
        self._position_log.clear()
        print(f"[PositionLog] Flushed {n} steps → {self._csv_path}  "
              f"(total_steps={self._total_steps})")

    def close(self):
        """env.close() 时把剩余缓冲也写进去"""
        self._flush()
        super().close()

    def rand_act(self):
        a = self.action_space.sample()
        return torch.as_tensor(a, dtype=torch.float32)


# ---------------------------
# CAG 环境入口：从 Hydra cfg 构建 env，并补齐 Trainer 需要的字段
# ---------------------------
def make_env(cfg):
    env = make_cag_env(cfg.env)

    log_dir = os.path.join(cfg.work_dir, "position_logs")
    env = LegacyStepAPIWrapper(env, log_dir=log_dir, flush_every=1000)
    return env


def _infer_shapes_for_cfg(cfg, env):
    """
    你们 parse_cfg 可能已经会推导这些字段；这里兜底补齐：
      obs_shape, action_dim, episode_length, seed_steps
    """
    # action_dim
    if getattr(cfg, "action_dim", None) in (None, "???"):
        if hasattr(env.action_space, "shape") and env.action_space.shape is not None:
            cfg.action_dim = int(np.prod(env.action_space.shape))
        else:
            raise ValueError("Cannot infer action_dim from env.action_space")

    # obs_shape：如果是 Dict space，建议你在环境里 already flatten；否则这里尽量推断
    if getattr(cfg, "obs_shape", None) in (None, "???"):
        ospec = env.observation_space
        if isinstance(ospec, gym.spaces.Box):
            cfg.obs_shape = tuple(ospec.shape)
        elif isinstance(ospec, gym.spaces.Dict):
            # dict -> flatten dim（按键顺序）
            dim = 0
            for sp in ospec.spaces.values():
                if isinstance(sp, gym.spaces.Box):
                    dim += int(np.prod(sp.shape))
                else:
                    raise ValueError("Dict observation contains non-Box space; please flatten in env.")
            cfg.obs_shape = (dim,)
        else:
            raise ValueError(f"Unsupported observation_space for obs_shape: {type(ospec)}")

    # episode_length：优先用你环境里 MaxSteps（cfg.env.max_steps）
    if getattr(cfg, "episode_length", None) in (None, "???"):
        if hasattr(cfg, "env") and getattr(cfg.env, "max_steps", None) is not None:
            cfg.episode_length = int(cfg.env.max_steps)
        elif hasattr(env, "max_episode_steps"):
            cfg.episode_length = int(env.max_episode_steps)
        else:
            # 兜底：不写也行，但有的代码会用
            cfg.episode_length = 200

    # seed_steps：很多实现默认用 episode_length * 5 或一个常数
    if getattr(cfg, "seed_steps", None) in (None, "???"):
        cfg.seed_steps = int(min(5000, cfg.episode_length * 10))

    # multitask：CAG 一般单任务
    if getattr(cfg, "multitask", None) in (None, "???"):
        cfg.multitask = False

    return cfg


@hydra.main(config_name="config", config_path=".")
def train(cfg: dict):
    """
    单任务 TD-MPC2 训练入口（适配 CAG_eve 环境）。
    """
    assert torch.cuda.is_available()
    assert cfg.steps > 0, "Must train for at least 1 step."

    cfg = parse_cfg(cfg)         # 你们原来的 cfg 处理（会设置 work_dir 等）
    set_seed(cfg.seed)

    print(colored("Work dir:", "yellow", attrs=["bold"]), cfg.work_dir)

    env = make_env(cfg)
    obs = env.reset()

    print("obs type:", type(obs))       # dict
    for k, v in obs.items():
        if hasattr(v, "shape"):
            print(f"{k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"{k}: type={type(v)}, value={v}")

    cfg = _infer_shapes_for_cfg(cfg, env)

    trainer_cls = OfflineTrainer if cfg.multitask else OnlineTrainer

    trainer = trainer_cls(
        cfg=cfg,
        env=env,
        agent=TDMPC2(cfg),
        buffer=Buffer(cfg),
        logger=Logger(cfg),
    )
    trainer.train()
    print("\nTraining completed successfully")
    env.close()  # 触发最后一次 flush


if __name__ == "__main__":
    train()
