from time import time
import numpy as np
import torch
from tensordict.tensordict import TensorDict
from .base import Trainer
from tqdm import tqdm
from typing import Any, Dict, Union


def _to_torch_any(x: Any, device) -> Union[torch.Tensor, Dict[str, Any]]:
	if isinstance(x, dict):
		return {k: _to_torch_any(v, device) for k, v in x.items()}

	if isinstance(x, np.ndarray):
		x = torch.from_numpy(x)
	elif not torch.is_tensor(x):
		x = torch.as_tensor(x)

	# 到这里，x 一定是 Tensor
	return x.to(device=device, dtype=torch.float32)

class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		self.device = self.agent.device

	# 通用指标：训练 / 测试都会用到
	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		elapsed_time = time() - self._start_time
		return dict(
			step=self._step,		# 当前总 step 数
			episode=self._ep_idx,		# 当前 episode 编号
			elapsed_time=elapsed_time,		# 已运行时间
			steps_per_second=self._step / elapsed_time		# 吞吐率：每秒多少 step
		)

	# 评估函数（不更新模型，只 rollout）
	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		# 用于保存每个 episode 的 reward / success / length
		ep_rewards, ep_successes, ep_lengths = [], [], []
		for i in range(self.cfg.eval_episodes):		# 进行eval_episodes次的验证
			# 重置环境，done=False，累计 reward=0，时间步 t=0
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			# 如果需要保存视频
			if self.cfg.save_video:
				# 只在第一个 episode 启用视频
				self.logger.video.init(self.env, enabled=(i==0))
			# rollout 一个 episode
			while not done:
				# print("EVAL step", t)
				torch.compiler.cudagraph_mark_step_begin()		# CUDA graph 的 step 标记（用于编译优化）
				action = self.agent.act(obs, t0=t==0, eval_mode=True)		# agent 选择动作（eval_mode=True，不加噪声）

				obs, reward, done, info = self.env.step(action)		# 环境执行动作
				# print(f'action={action}, obs={obs}, reward={reward}, done={done}, info={info}')
				# env0 = self.env.unwrapped  # 拿到最底层原始 env
				# print(f'interim_target={env0.interim_target.reached}')
				# print(f'step reward: {reward}')
				ep_reward += reward		# 累积 reward
				# print(f'ep reward: {ep_reward}')
				t += 1		# 时间步 +1
				if self.cfg.save_video:		# 记录视频帧
					self.logger.video.record(self.env)
			# 保存该 episode 的统计量
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			ep_lengths.append(t)
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		# 返回 eval 的平均结果
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
			episode_length= np.nanmean(ep_lengths),
		)

	@staticmethod
	def _to_torch(x, device):
		if isinstance(x, dict):
			x = x.get("state", x)  # 你现在 obs 通常是 {"state": ...}
		# print(f'x={x}')
		if isinstance(x, np.ndarray):
			x = torch.from_numpy(x)
		if not torch.is_tensor(x):
			x = torch.as_tensor(x)
		return x.to(device=device, dtype=torch.float32)



	# 将一次 transition 转换为 TensorDict
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

	# 主训练循环
	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, True, False

		# 🔹 初始化进度条（从当前 step 开始，支持 resume）
		pbar = tqdm(
			initial=self._step,
			total=self.cfg.steps,
			desc="TD-MPC2 Training",
			dynamic_ncols=True,
		)

		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			if self._step > 0 and self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Reset environment
			if done:
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0:
					if info['terminated'] and not self.cfg.episodic:
						raise ValueError(
							'Termination detected but you are not in episodic mode. '
							'Set `episodic=true` to enable support for terminations.'
						)

					train_metrics.update(
						episode_reward=torch.tensor(
							[td['reward'] for td in self._tds[1:]]
						).sum(),
						episode_success=info['success'],
						episode_length=len(self._tds),
						episode_terminated=info['terminated'],
					)
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					print(type(self._tds[0]["obs"]), self._tds[0]["obs"].keys() if isinstance(self._tds[0]["obs"], dict) else "")
					self._ep_idx = self.buffer.add(self._tds)

				obs = self.env.reset()
				self._tds = [self.to_td(obs)]

			# === 与环境交互 ===
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds) == 1)
			else:
				action = self.env.rand_act()

			obs, reward, done, info = self.env.step(action)

			# === A: 避免把“仿真错误步”写进 _tds / replay ===
			sim_err = bool(info.get("simulation_error", False))
			obs_has_nan = False
			try:
				# 兼容 obs 是 dict/np/torch
				if isinstance(obs, dict):
					for v in obs.values():
						if hasattr(v, "detach"):  # torch
							vv = v.detach()
							obs_has_nan |= torch.isnan(vv).any().item() or torch.isinf(vv).any().item()
						else:  # numpy
							obs_has_nan |= (np.isnan(v).any() or np.isinf(v).any())
				else:
					if hasattr(obs, "detach"):
						vv = obs.detach()
						obs_has_nan = torch.isnan(vv).any().item() or torch.isinf(vv).any().item()
					else:
						obs_has_nan = np.isnan(obs).any() or np.isinf(obs).any()
			except Exception:
				# 检查失败就按异常处理（宁可丢掉也别污染）
				obs_has_nan = True

			if sim_err or obs_has_nan:
				# 关键：不要 append 这个 transition
				# 关键：强制结束 episode，让下一轮 reset
				done = True
				info["terminated"] = False
				info["truncated"] = True
				reward = -1.0  # 保底
				self._sim_err_cnt = getattr(self, "_sim_err_cnt", 0) + 1
				pbar.set_postfix({"sim_err": self._sim_err_cnt, "reward": f"{reward:.2f}"})

			# 可选：给一个明确惩罚（看你 reward 设计）
			# reward = -1.0
			else:
				self._tds.append(self.to_td(obs, action, reward, done, info))

			# === 更新 agent ===
			if self._step >= self.cfg.seed_steps:
				if len(self.buffer._episodes) == 0:
					self._step += 1
					pbar.update(1)
					continue

				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					print('Pretraining agent on seed data...')
				else:
					num_updates = 1

				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer)
				train_metrics.update(_train_metrics)

				# 🔹 在进度条里显示关键信息（可选）
				if 'loss' in _train_metrics:
					pbar.set_postfix({
						"loss": f"{_train_metrics['loss']:.3f}",
						"reward": f"{reward:.2f}"
					})

			self._step += 1
			pbar.update(1)

			if self._step % self.cfg.save_freq == 0:
				try:
					self.logger.save_agent(self.agent, identifier=f"step_{self._step}")
				except Exception as e:
					print(f"Checkpoint save failed: {e}")

		pbar.close()
		self.logger.finish(self.agent)
