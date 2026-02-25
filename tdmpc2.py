import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from common.layers import api_model_conversion
from tensordict import TensorDict
import numpy as np


class TDMPC2(torch.nn.Module):
	"""
	TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and multi-task experiments,
	and supports both state and pixel observations.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device('cuda:0')
		self.model = WorldModel(cfg).to(self.device)
		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._termination.parameters() if self.cfg.episodic else []},
			{'params': self.model._Qs.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []
			 }
		], lr=self.cfg.lr, capturable=True)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		print('Episode length:', cfg.episode_length)
		print('Discount factor:', self.discount)
		self.register_buffer(
			"_prev_mean",
			torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device),
			persistent=False,
		)

		if cfg.compile:
			print('Compiling update function with torch.compile...')
			self._update = torch.compile(self._update, mode="reduce-overhead")

	@property
	def plan(self):
		_plan_val = getattr(self, "_plan_val", None)
		if _plan_val is not None:
			return _plan_val
		if self.cfg.compile:
			plan = torch.compile(self._plan, mode="reduce-overhead")
		else:
			plan = self._plan
		self._plan_val = plan
		return self._plan_val

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict()}, fp)


	def load(self, path):
		ckpt = torch.load(path, map_location=self.device)
		state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

		if "model" in state_dict and isinstance(state_dict["model"], dict):
			state_dict = state_dict["model"]

		try:
			self.model.load_state_dict(state_dict, strict=True)
			return
		except Exception as e:
			print(f"[load] direct strict load failed: {e}")

		try:
			self.model.load_state_dict(state_dict, strict=False)
			print("[load] loaded with strict=False (some keys missing/unexpected).")
			return
		except Exception as e:
			print(f"[load] direct non-strict load failed: {e}")

		state_dict2 = api_model_conversion(self.model.state_dict(), state_dict)
		self.model.load_state_dict(state_dict2, strict=False)
		print("[load] loaded via api_model_conversion.")

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (int): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		if isinstance(obs, dict):
			obs = _to_torch_any(obs, self.device)
			# add batch dim for each field
			obs = {k: (v.unsqueeze(0) if v.dim() in (1, 2) else v) for k, v in obs.items()}
		else:
			obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)

		if task is not None:
			task = torch.tensor([task], device=self.device)
		if self.cfg.mpc:
			return self.plan(obs, t0=t0, eval_mode=eval_mode, task=task).cpu()
		z = self.model.encode(obs, task)
		action, info = self.model.pi(z, task)
		if eval_mode:
			action = info["mean"]
		return action[0].cpu()

	@torch.no_grad()
	def _estimate_value(self, z, actions, task):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		# print(f'z shape = {z.shape}')		# [512, 768]
		# actions shape [3 512 2]
		G, discount = 0, 1
		termination = torch.zeros(self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
			z = self.model.next(z, actions[t], task)
			G = G + discount * (1-termination) * reward
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
			if self.cfg.episodic:
				termination = torch.clip(termination + (self.model.termination(z, task) > 0.5).float(), max=1.)
		action, _ = self.model.pi(z, task)
		# print(f'_estimate_value a shape = {action.shape}')		# [512 2]
		return G + discount * (1-termination) * self.model.Q(z, action, task, return_type='avg')

	@torch.no_grad()
	def _plan(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
			z (torch.Tensor): Latent state from which to plan.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		# Sample policy trajectories
		z = self.model.encode(obs, task)
		# print(f'z shape = {z.shape}')		# [1, 768]
		if self.cfg.num_pi_trajs > 0:
			# pi_actions: (horizon, num_pi_trajs, action_dim)
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon-1):
				pi_actions[t], _ = self.model.pi(_z, task)
				_z = self.model.next(_z, pi_actions[t], task)
			pi_actions[-1], _ = self.model.pi(_z, task)


		z = z.repeat(self.cfg.num_samples, 1)		# [512, 768]
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions
			
		for _ in range(self.cfg.iterations):
			r = torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
			actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions = torch.cat([actions[:, :self.cfg.num_pi_trajs, :], actions_sample], dim=1)
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			value = self._estimate_value(z, actions, task).nan_to_num(0)	# [1, 512, 1]
			v = value[:, 0]  # [512,1] -> [512]
			K = min(int(self.cfg.num_elites), int(v.numel()))
			elite_idxs = torch.topk(v, K, dim=0).indices	# 0-63

			elite_idxs = torch.topk(v, K, dim=0).indices
			elite_value = v[elite_idxs]
			elite_actions = actions[:, elite_idxs, :]
			# print(f'elite_actions={elite_actions.shape}')		# [3, 64, 2]

			# Update parameters
			max_value = elite_value.max(0).values
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score = score / score.sum(0)
			# print(f'score={score.shape}')		# [64]
			w = score.view(1, -1, 1)  # [1, 64, 1]
			mean = (w * elite_actions).sum(dim=1)  # [H, A]
			var = (w * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1)  # [H, A]
			std = (var + 1e-9).sqrt().clamp(self.cfg.min_std, self.cfg.max_std)
			std = std.clamp(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		# Select action
		rand_idx = math.gumbel_softmax_sample(score)
		actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
		a, std = actions[0], std[0]
		if not eval_mode:
			a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
		self._prev_mean.copy_(mean)
		return a.clamp(-1, 1)

	def update_pi(self, zs, task):
		action, info = self.model.pi(zs, task)  # action: [T,B,A]
		T = zs.shape[0] if zs.dim() == 3 else 1

		if zs.dim() == 2:
			zs = zs.unsqueeze(0)
			action = action.unsqueeze(0)
			if info["scaled_entropy"].dim() == 1:
				info["scaled_entropy"] = info["scaled_entropy"].unsqueeze(0)

		T, B, Z = zs.shape
		A = action.shape[-1]
		z2 = zs.reshape(T * B, Z)  # [T*B, Z]
		a2 = action.reshape(T * B, A)  # [T*B, A]

		qs2 = self.model.Q(z2, a2, task, return_type='avg', detach=True)  # [T*B]
		qs = qs2.view(T, B)  # [T, B]

		self.scale.update(qs[0])  # [B]
		qs = self.scale(qs)  
		ent = info["scaled_entropy"]
		if ent.dim() == 3 and ent.size(-1) == 1:
			ent = ent.squeeze(-1)
		if ent.dim() == 1:
			ent = ent.unsqueeze(0).expand(T, -1)

		obj = self.cfg.entropy_coef * ent + qs  # [T,B]
		obj_t = obj.mean(dim=1)  # [T]

		rho = torch.pow(self.cfg.rho, torch.arange(T, device=self.device, dtype=obj_t.dtype))  # [T]
		pi_loss = (-(obj_t) * rho).mean()

		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		pi_info = {
			"pi_loss": pi_loss.detach(),
			"pi_grad_norm": pi_grad_norm.detach() if torch.is_tensor(pi_grad_norm) else pi_grad_norm,
		}
		return pi_loss, pi_info

	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			terminated (torch.Tensor): Termination signal at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: TD-target.
		"""
		action, _ = self.model.pi(next_z, task)
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * (1-terminated) * self.model.Q(next_z[0], action[0], task, return_type='min', target=True)


	def _update(self, obs, action, reward, terminated, task=None):
		# print(1)
		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(_slice_obs(obs, slice(1, None)), task)
			# 计算 td_targets
			td_targets = self._td_target(next_z, reward, terminated, task)

		# Prepare for update
		self.model.train()

		# zs: (horizon+1, batch, latent_dim)
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		# z0 = encode(obs[0])
		z = self.model.encode(_slice_obs(obs, 0), task)
		zs[0] = z
		consistency_loss = 0
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			z = self.model.next(z, _action, task)
			consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
			zs[t+1] = z

		_zs = zs[:-1]
		# print(f'update a shape = {action.shape}')
		qs = self.model.Q(_zs[0], action[0], task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task)
		if self.cfg.episodic:
			termination_pred = self.model.termination(zs[1:], task, unnormalized=True)

		# Compute losses
		reward_loss, value_loss = 0, 0

		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

		consistency_loss = consistency_loss / self.cfg.horizon
		reward_loss = reward_loss / self.cfg.horizon
		if self.cfg.episodic:
			termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated)
		else:
			termination_loss = 0.
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.termination_coef * termination_loss +
			self.cfg.value_coef * value_loss
		)

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		# Update policy
		pi_loss, pi_info = self.update_pi(zs.detach(), task)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		info = {
			"consistency_loss": consistency_loss.detach(),
			"reward_loss": reward_loss.detach(),
			"value_loss": value_loss.detach(),
			"termination_loss": termination_loss.detach() if torch.is_tensor(termination_loss) else termination_loss,
			"total_loss": total_loss.detach(),
			"grad_norm": grad_norm.detach() if torch.is_tensor(grad_norm) else grad_norm,
		}

		if self.cfg.episodic:
			info.update(math.termination_statistics(torch.sigmoid(termination_pred[-1]), terminated[-1]))
		info.update(pi_info)
		info["pi_loss_scalar"] = pi_loss.detach()  
		if torch.rand(()) < 0.001:
			print(
				"reward mean/std:",
				reward.mean().item(), reward.std().item(),
				"td mean/std:",
				td_targets.mean().item(), td_targets.std().item(),
			)

		return info

	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		#print(f'start update')
		obs, action, reward, terminated, task = buffer.sample()
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		torch.compiler.cudagraph_mark_step_begin()

		return self._update(obs, action, reward, terminated, **kwargs)

def _to_torch_any(x, device):
	import numpy as np
	import torch
	if isinstance(x, dict):
		return {k: _to_torch_any(v, device) for k, v in x.items()}
	if isinstance(x, np.ndarray):
		x = torch.from_numpy(x)
	if not torch.is_tensor(x):
		x = torch.as_tensor(x)
	return x.to(device=device, dtype=torch.float32)

def _slice_obs(obs, sl):
	# sl can be slice or int
	if isinstance(obs, dict):
		return {k: v[sl] for k, v in obs.items()}
	return obs[sl]
