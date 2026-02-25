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

	return x.to(device=device, dtype=torch.float32)

class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		self.device = self.agent.device

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		elapsed_time = time() - self._start_time
		return dict(
			step=self._step,		
			episode=self._ep_idx,		
			elapsed_time=elapsed_time,		
			steps_per_second=self._step / elapsed_time		
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes, ep_lengths = [], [], []
		for i in range(self.cfg.eval_episodes):		
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				# print("EVAL step", t)
				torch.compiler.cudagraph_mark_step_begin()		
				action = self.agent.act(obs, t0=t==0, eval_mode=True)		

				obs, reward, done, info = self.env.step(action)		
				ep_reward += reward		
				# print(f'ep reward: {ep_reward}')
				t += 1		
				if self.cfg.save_video:		
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			ep_lengths.append(t)
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
			episode_length= np.nanmean(ep_lengths),
		)

	@staticmethod
	def _to_torch(x, device):
		if isinstance(x, dict):
			x = x.get("state", x)  
		# print(f'x={x}')
		if isinstance(x, np.ndarray):
			x = torch.from_numpy(x)
		if not torch.is_tensor(x):
			x = torch.as_tensor(x)
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

		if done is not None:
			data["done"] = torch.as_tensor(done, device=self.device, dtype=torch.bool).reshape(())

		data["info"] = info

		terminated = None
		truncated = None
		if info is not None and isinstance(info, dict):
			if "terminated" in info:
				terminated = bool(info["terminated"])
				data["terminated"] = torch.as_tensor(terminated, device=self.device, dtype=torch.bool).reshape(())
			if "truncated" in info:
				truncated = bool(info["truncated"])
				data["truncated"] = torch.as_tensor(truncated, device=self.device, dtype=torch.bool).reshape(())

		if done is None and (terminated is not None or truncated is not None):
			d = (terminated is True) or (truncated is True)
			data["done"] = torch.as_tensor(d, device=self.device, dtype=torch.bool).reshape(())

		done_for_learning = None
		if info is not None and isinstance(info, dict) and "done_for_learning" in info:
			done_for_learning = bool(info["done_for_learning"])
		elif terminated is not None:
			done_for_learning = bool(terminated)
		elif done is not None:
			done_for_learning = bool(done)
		else:
			done_for_learning = False

		data["done_for_learning"] = torch.as_tensor(done_for_learning, device=self.device, dtype=torch.bool).reshape(())

		return data

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, True, False

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

			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds) == 1)
			else:
				action = self.env.rand_act()

			obs, reward, done, info = self.env.step(action)

			sim_err = bool(info.get("simulation_error", False))
			obs_has_nan = False
			try:
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
				obs_has_nan = True

			if sim_err or obs_has_nan:
				done = True
				info["terminated"] = False
				info["truncated"] = True
				reward = -1.0  
				self._sim_err_cnt = getattr(self, "_sim_err_cnt", 0) + 1
				pbar.set_postfix({"sim_err": self._sim_err_cnt, "reward": f"{reward:.2f}"})

			# reward = -1.0
			else:
				self._tds.append(self.to_td(obs, action, reward, done, info))

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
