import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
import warnings
warnings.filterwarnings('ignore')

from typing import Any, Dict, Optional
import hydra
import imageio
import numpy as np
import torch
from termcolor import colored
import gymnasium as gym
import cv2
from typing import Any, Dict, Union

from common.parser import parse_cfg
from common.seed import set_seed
from CAG_eve.env import make_eval_env
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True

import os
print("MAIN PID =", os.getpid())


class LegacyStepAPIWrapper(gym.Wrapper):
    """
    将 gymnasium 风格：
        reset() -> (obs, info)
        step()  -> (obs, reward, terminated, truncated, info)
    适配成你们 OnlineTrainer 里用的旧风格：
        reset() -> obs
        step()  -> (obs, reward, done, info)
    并确保 info 里有：
        info['terminated'] : bool
        info['success']    : bool (尽量推断/默认 False)
    以及提供 rand_act() 给 Trainer 用。
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._last_info: Dict[str, Any] = {}

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            # 兼容一些非标准 env
            obs, info = out, {}
        self._last_info = dict(info) if isinstance(info, dict) else {}
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
            # 兼容旧 env: obs, reward, done, info
            obs, reward, done, info = out
            info = dict(info) if isinstance(info, dict) else {}
            info.setdefault("terminated", bool(done))
            info.setdefault("truncated", False)
            info.setdefault("done_for_learning", bool(done))
        else:
            raise RuntimeError(f"Unexpected env.step return: {type(out)} / len={len(out) if isinstance(out, tuple) else 'NA'}")

        # success：你的 eve.info 里可能已有 success；没有就默认 False
        info.setdefault("success", bool(info.get("is_success", False)))
        info.setdefault("is_success", bool(info.get("success", False)))

        self._last_info = info
        return obs, float(reward), bool(done), info

    def rand_act(self):
        # 你们 Trainer 里会用 self.env.rand_act()
        a = self.action_space.sample()
        # 一些实现会用 torch tensor；你的 Trainer 在 to_td 里用 torch.full_like(self.env.rand_act())
        # 所以这里返回 torch.Tensor 更稳
        return torch.as_tensor(a, dtype=torch.float32)


def make_env(cfg):
    """
    cfg 来自 Hydra/parse_cfg，要求 cfg.env 下有 CAG 环境参数：
      cfg.env.model_folder / insertion_vessel_name / insertion_point_idx / ...
    """
    # 你自己的 make_cag_env 接收的是“属性访问”的 cfg（cfg.xxx）
    # Hydra cfg.env 是 OmegaConf 的节点，也支持点访问，直接传即可
    env = make_eval_env(cfg.env)

    # 适配你们 Trainer 的旧接口
    env = LegacyStepAPIWrapper(env)

    # （可选）如果你们 Logger/video 需要 render，也可以在这里再包一层，或让 env.render() 可用
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

def _to_torch_any(x: Any, device) -> Union[torch.Tensor, Dict[str, Any]]:
	if isinstance(x, dict):
		return {k: _to_torch_any(v, device) for k, v in x.items()}

	if isinstance(x, np.ndarray):
		x = torch.from_numpy(x)
	elif not torch.is_tensor(x):
		x = torch.as_tensor(x)

	# 到这里，x 一定是 Tensor
	return x.to(device=device, dtype=torch.float32)


def to_td(self, obs, action=None, reward=None, done=None, info=None):
    data = {}
    obs_t = _to_torch_any(obs, self.device)  # dict -> dict of tensors
    data["obs"] = obs_t

    if action is not None:
        a = self._to_torch(action, self.device).float()
        data["action"] = a.reshape(-1)

    if reward is not None:
        data["reward"] = torch.as_tensor(reward, device=self.device, dtype=torch.float32).reshape(())

    # =========================
    # 1) 环境层 done：用于 reset（terminated or truncated）
    # =========================
    if done is not None:
        data["done"] = torch.as_tensor(done, device=self.device, dtype=torch.bool).reshape(())

    data["info"] = info

    # =========================
    # 2) 解析 terminated / truncated
    # =========================
    terminated = None
    truncated = None
    if info is not None and isinstance(info, dict):
        if "terminated" in info:
            terminated = bool(info["terminated"])
            data["terminated"] = torch.as_tensor(terminated, device=self.device, dtype=torch.bool).reshape(())
        if "truncated" in info:
            truncated = bool(info["truncated"])
            data["truncated"] = torch.as_tensor(truncated, device=self.device, dtype=torch.bool).reshape(())

    # 如果外部没传 done，就根据 (terminated or truncated) 补齐 done（仅用于 reset 语义）
    if done is None and (terminated is not None or truncated is not None):
        d = (terminated is True) or (truncated is True)
        data["done"] = torch.as_tensor(d, device=self.device, dtype=torch.bool).reshape(())

    # =========================
    # 3) ✅ 学习层 done：只把 terminated 当“真终止”
    #    优先用 wrapper 写入的 done_for_learning
    # =========================
    done_for_learning = None
    if info is not None and isinstance(info, dict) and "done_for_learning" in info:
        done_for_learning = bool(info["done_for_learning"])
    elif terminated is not None:
        done_for_learning = bool(terminated)
    elif done is not None:
        # 兜底：旧 env 没区分，就只能用 done
        done_for_learning = bool(done)
    else:
        done_for_learning = False

    data["done_for_learning"] = torch.as_tensor(done_for_learning, device=self.device, dtype=torch.bool).reshape(())

    return data


@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
    """
    Script for evaluating a single-task / multi-task TD-MPC2 checkpoint.

    Most relevant args:
        `task`: task name (or mt30/mt80 for multi-task evaluation)
        `model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
        `checkpoint`: path to model checkpoint to load
        `eval_episodes`: number of episodes to evaluate on per task (default: 10)
        `save_video`: whether to save a video of the evaluation (default: True)
        `seed`: random seed (default: 1)

    See config.yaml for a full list of args.

    Example usage:
    ```
    $ python evaluate.py task=mt80 model_size=48 checkpoint=/path/to/mt80-48M.pt
    $ python evaluate.py task=mt30 model_size=317 checkpoint=/path/to/mt30-317M.pt
    $ python evaluate.py task=dog-run checkpoint=/path/to/dog-1.pt save_video=true
    ```
    """
    assert torch.cuda.is_available()
    assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
    print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
    print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))

    if not cfg.multitask and ('mt80' in cfg.checkpoint or 'mt30' in cfg.checkpoint):
        print(colored(
            'Warning: single-task evaluation of multi-task models is not currently supported.',
            'red', attrs=['bold']
        ))
        print(colored(
            'To evaluate a multi-task model, use task=mt80 or task=mt30.',
            'red', attrs=['bold']
        ))

    # Make environment
    env = make_env(cfg)
    cfg = _infer_shapes_for_cfg(cfg, env)

    # Load agent
    agent = TDMPC2(cfg)
    assert os.path.exists(cfg.checkpoint), \
        f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
    agent.load(cfg.checkpoint)

    # Evaluate
    if cfg.multitask:
        print(colored(
            f'Evaluating agent on {len(cfg.tasks)} tasks:',
            'yellow', attrs=['bold']
        ))
    else:
        print(colored(
            f'Evaluating agent on {cfg.task}:',
            'yellow', attrs=['bold']
        ))

    video_dir = os.path.join(cfg.work_dir, 'videos')
    print(f'video dir: {video_dir}')
    os.makedirs(video_dir, exist_ok=True)

    scores = []
    tasks = cfg.tasks if cfg.multitask else [cfg.task]

    for task_idx, task in enumerate(tasks):
        if not cfg.multitask:
            task_idx = None

        ep_rewards, ep_successes = [], []

        for i in range(cfg.eval_episodes):
            obs, done, ep_reward, t = env.reset(), False, 0, 0
            print(f'obs={obs}')

            frames = [env.render()]

            while not done:
                print(f't={t}')
                action = agent.act(obs, t0=t == 0, task=task_idx)
                obs, reward, done, info = env.step(action)
                ep_reward += reward
                t += 1

                frames.append(env.render())

            ep_rewards.append(ep_reward)
            ep_successes.append(info['success'])

            print(f"帧数量: {len(frames)}")
            if frames:
                print(
                    f"数据类型: {frames[0].dtype}, 尺寸: {frames[0].shape}, 最小值: {frames[0].min()}, 最大值: {frames[0].max()}")

            imageio.mimsave(
                os.path.join(video_dir, f'{task}-{i}.mp4'),
                frames,
                fps=15
            )

        ep_rewards = np.mean(ep_rewards)
        ep_successes = np.mean(ep_successes)

        if cfg.multitask:
            scores.append(
                ep_successes * 100 if task.startswith('mw-') else ep_rewards / 10
            )

        print(colored(
            f'  {task:<22}\tR: {ep_rewards:.01f}  \tS: {ep_successes:.02f}',
            'yellow'
        ))

    if cfg.multitask:
        print(colored(
            f'Normalized score: {np.mean(scores):.02f}',
            'yellow', attrs=['bold']
        ))


if __name__ == '__main__':
    evaluate()
