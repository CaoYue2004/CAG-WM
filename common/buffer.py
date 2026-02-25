import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def _to_torch(x, dtype=torch.float32):
    if isinstance(x, dict):
        return {k: _to_torch(v, dtype=dtype) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    return x.to(dtype=dtype)


class Buffer:
    """
    Replay buffer for TD-MPC2 training (NO torchrl / NO tensordict).
    Stores full episodes, samples horizon+1 subsequences like SliceSampler(strict_length=True).

    sample() returns:
        obs:        [H+1, B, obs_dim]
        action:     [H,   B, act_dim]    (aligned with original: td['action'][1:])
        reward:     [H,   B, 1]          (aligned with original: td['reward'][1:])
        terminated: [H,   B, 1]          (aligned with original: td['terminated'][1:], or zeros)
        task:       [B, task_dim] or None
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._capacity = min(int(cfg.buffer_size), int(cfg.steps))
        self._num_eps = 0

        self._episodes: List[Dict[str, torch.Tensor]] = []

    @property
    def capacity(self):
        return self._capacity

    @property
    def num_eps(self):
        return self._num_eps
    def load(self, episodes: List[Dict[str, torch.Tensor]]):
        """
        episodes: list of episode dicts in the internal format:
            {"obs": [T+1,D], "action":[T,A], "reward":[T], "terminated":[T] (optional), "task":[task_dim] (optional)}
        """
        for ep in episodes:
            self._append_episode(ep)
        return self._num_eps
    def add(self, tds: Any):
        if isinstance(tds, list):
            ep = self._pack_episode_from_tds(tds)
        elif isinstance(tds, dict) and "obs" in tds:
            ep = tds
        else:
            raise TypeError(f"Unsupported add() input type: {type(tds)}")

        if ep is None:
            return self._num_eps
        self._append_episode(ep)
        ep = self._episodes[-1]
        # print("EP keys:", ep.keys())
        return self._num_eps

    # --------- internal helpers ----------
    def _append_episode(self, ep: Dict[str, torch.Tensor]):
        if len(self._episodes) >= self._capacity:
            self._episodes.pop(0)
        self._episodes.append(ep)
        self._num_eps += 1

    def _pack_episode_from_tds(self, tds: List[Dict[str, Any]]) -> Optional[Dict[str, torch.Tensor]]:
        H = int(self.cfg.horizon)

        if not tds or "obs" not in tds[0]:
            return None
        obs0 = _to_torch(tds[0]["obs"])  # dict
        obs_list: List[Dict[str, torch.Tensor]] = [obs0]

        act_list: List[torch.Tensor] = []
        rew_list: List[float] = []
        done_learn_list: List[float] = []
        task_val: Optional[torch.Tensor] = None

        for td in tds[1:]:
            if "obs" not in td or "action" not in td or "reward" not in td:
                continue

            o = _to_torch(td["obs"])  # dict
            a = _to_torch(td["action"]).squeeze(0)  # [act_dim]

            r = td["reward"]
            if torch.is_tensor(r):
                r = float(r.detach().cpu().view(-1)[0].item())
            else:
                r = float(r)

            info = td.get("info", None)

            done_for_learning = False
            if "done_for_learning" in td:
                done_for_learning = bool(td["done_for_learning"])
            elif isinstance(info, dict) and "done_for_learning" in info:
                done_for_learning = bool(info["done_for_learning"])
            elif isinstance(info, dict) and "terminated" in info:
                done_for_learning = bool(info["terminated"])
            elif "terminated" in td:
                done_for_learning = bool(td["terminated"])
            elif "done" in td:
                done_for_learning = bool(td["done"])

            if task_val is None:
                if "task" in td:
                    task_val = _to_torch(td["task"]).view(-1)  
                elif isinstance(info, dict) and "task" in info:
                    task_val = _to_torch(info["task"]).view(-1)

            obs_list.append(o)
            act_list.append(a)
            rew_list.append(r)
            done_learn_list.append(float(done_for_learning))

        T = len(act_list)
        if T < H:
            return None

        keys = obs_list[0].keys()
        obs = {k: torch.stack([o[k] for o in obs_list], dim=0) for k in keys}
        action = torch.stack(act_list, dim=0)  # [T, act_dim]
        reward = torch.tensor(rew_list, dtype=torch.float32).view(T)  # [T]
        done_for_learning = torch.tensor(done_learn_list, dtype=torch.float32).view(T)  # [T]

        ep: Dict[str, Any] = {
            "obs": {k: v.cpu() for k, v in obs.items()},
            "action": action.cpu(),
            "reward": reward.cpu(),
            "done_for_learning": done_for_learning.cpu(),
            # "terminated": done_for_learning.cpu(),
        }

        if task_val is not None:
            ep["task"] = task_val.cpu()
        else:
            ep["task"] = None  

        return ep

    def _prepare_batch(self, obs, action, reward, terminated, task):
        if isinstance(obs, dict):
            obs = {k: v.to(self._device, non_blocking=True).contiguous() for k, v in obs.items()}
        else:
            obs = obs.to(self._device, non_blocking=True).contiguous()
        action = action.to(self._device, non_blocking=True).contiguous()
        reward = reward.to(self._device, non_blocking=True).unsqueeze(-1).contiguous()          # [H,B,1]
        terminated = terminated.to(self._device, non_blocking=True).unsqueeze(-1).contiguous()  # [H,B,1]

        if task is not None:
            task = task.to(self._device, non_blocking=True).contiguous()
        return obs, action, reward, terminated, task

    def sample(self):
        assert len(self._episodes) > 0, "Buffer is empty"

        B = int(self.cfg.batch_size)
        H = int(self.cfg.horizon)

        obs_batch = []
        act_batch = []
        rew_batch = []
        term_batch = []
        task_batch = []

        for _ in range(B):
            ep = random.choice(self._episodes)
            obs_seq = ep["obs"]               # [T+1,D]
            act_seq = ep["action"]            # [T,A]
            rew_seq = ep["reward"]            # [T]
            term_seq = ep.get("done_for_learning", ep.get("terminated", None))  # [T] or None
            if term_seq is None:
                term_seq = torch.zeros_like(rew_seq)

            T = act_seq.shape[0]
            start = random.randint(0, T - H)

            obs_clip = {k: v[start:start+H+1] for k, v in obs_seq.items()}
            act_clip = act_seq[start:start + H]              # [H,A]
            rew_clip = rew_seq[start:start + H]              # [H]
            term_clip = term_seq[start:start + H]            # [H]

            obs_batch.append(obs_clip)
            act_batch.append(act_clip)
            rew_batch.append(rew_clip)
            term_batch.append(term_clip)

            tv = ep.get("task", None)
            if tv is None:
                task_batch.append(None)
            else:
                task_batch.append(tv)

        keys = obs_batch[0].keys()
        obs = {}
        for k in keys:
            # stack -> [B, H+1, ...] then permute -> [H+1, B, ...]
            obs[k] = torch.stack([o[k] for o in obs_batch], dim=0).permute(1, 0, *range(2, obs_batch[0][k].dim() + 1))
        # print(f'act_batch len={len(act_batch)}, shape={act_batch[0].shape}')
        action = torch.stack(act_batch, dim=0).permute(1, 0, 2)                 # [H,B,A]
        reward = torch.stack(rew_batch, dim=0).permute(1, 0)                    # [H,B]
        terminated = torch.stack(term_batch, dim=0).permute(1, 0)               # [H,B]

        if all(t is None for t in task_batch):
            task = None
        else:
            first = next(t for t in task_batch if t is not None)
            task_dim = int(first.numel())
            task = torch.stack(
                [(t if t is not None else torch.zeros(task_dim)) for t in task_batch],
                dim=0
            )  # [B, task_dim]

        return self._prepare_batch(obs, action, reward, terminated, task)
