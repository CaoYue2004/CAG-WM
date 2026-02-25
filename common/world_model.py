from copy import deepcopy
import copy
import torch
import torch.nn as nn

from common import layers, math, init
from tensordict import TensorDict
from tensordict.nn import TensorDictParams


class WorldModel(nn.Module):
	"""
	TD-MPC2 implicit world model architecture.
	Can be used for both single-task and multi-task experiments.
	"""
	# TD-MPC2 的“隐式世界模型”总成：encoder + dynamics + reward + termination + policy prior + Q-ensemble

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		if cfg.multitask:
			self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
			self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim))
			for i in range(len(cfg.tasks)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.
		self._encoder = layers.enc(cfg)
		self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
		self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		self._termination = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 1) if cfg.episodic else None
		self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		self._Qs = layers.Ensemble([
			layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2 * [cfg.mlp_dim],
					   max(cfg.num_bins, 1), dropout=cfg.dropout)
			for _ in range(cfg.num_q)
		])
		self._target_Qs = layers.Ensemble([
			layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2 * [cfg.mlp_dim],
					   max(cfg.num_bins, 1), dropout=cfg.dropout)
			for _ in range(cfg.num_q)
		])

		self.apply(init.weight_init)
		# reward head last layer
		init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])

		self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
		self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
		self.init()

	def init(self):
		# 用 state_dict 同步权重（兼容所有情况）
		self._target_Qs.load_state_dict(self._Qs.state_dict())
		for p in self._target_Qs.parameters():
			p.requires_grad_(False)
		self._target_Qs.train(False)

	def __repr__(self):
		repr = 'TD-MPC2 World Model\n'
		modules = ['Encoder', 'Dynamics', 'Reward', 'Termination', 'Policy prior', 'Q-functions']
		for i, m in enumerate([self._encoder, self._dynamics, self._reward, self._termination, self._pi, self._Qs]):
			if m == self._termination and not self.cfg.episodic:
				continue
			repr += f"{modules[i]}: {m}\n"
		repr += "Learnable parameters: {:,}".format(self.total_params)
		return repr

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		# super().to 会搬动 self._Qs，但 deepcopy 出来的 _target_Qs 需要重新建/搬一次
		self.init()
		return self

	def train(self, mode=True):
		super().train(mode)
		self._target_Qs.train(False)
		return self

	@torch._dynamo.disable
	@torch.no_grad()
	def soft_update_target_Q(self):
		tau = float(self.cfg.tau)
		for p, tp in zip(self._Qs.parameters(), self._target_Qs.parameters()):
			tp.data.lerp_(p.data, tau)

	def task_emb(self, x, task):
		"""
		Continuous task embedding for multi-task experiments.
		Retrieves the task embedding for a given task ID `task`
		and concatenates it to the input `x`.
		"""
		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)
		emb = self._task_emb(task.long())
		if x.ndim == 3:
			emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif emb.shape[0] == 1:
			emb = emb.repeat(x.shape[0], 1)
		return torch.cat([x, emb], dim=-1)

	def encode(self, obs, task):
		"""
		Encodes an observation into its latent representation.
		This implementation assumes a single state-based observation.
		"""
		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		if self.cfg.obs == 'rgb' and obs.ndim == 5:
			return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
		return self._encoder[self.cfg.obs](obs)

	def next(self, z, a, task):
		"""
		Predicts the next latent state given the current latent state and action.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._dynamics(z)

	def reward(self, z, a, task):
		"""
		Predicts instantaneous (single-step) reward.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._reward(z)
	
	def termination(self, z, task, unnormalized=False):
		"""
		Predicts termination signal.
		"""
		assert task is None
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		if unnormalized:
			return self._termination(z)
		return torch.sigmoid(self._termination(z))
		

	def pi(self, z, task):
		"""
		Samples an action from the policy prior.
		The policy prior is a Gaussian distribution with
		mean and (log) std predicted by a neural network.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)

		# Gaussian policy prior
		mean, log_std = self._pi(z).chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mean)

		if self.cfg.multitask: # Mask out unused action dimensions
			mean = mean * self._action_masks[task]
			log_std = log_std * self._action_masks[task]
			eps = eps * self._action_masks[task]
			action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
		else: # No masking
			action_dims = None

		log_prob = math.gaussian_logprob(eps, log_std)

		# Scale log probability by action dimensions
		size = eps.shape[-1] if action_dims is None else action_dims
		scaled_log_prob = log_prob * size

		# Reparameterization trick
		action = mean + eps * log_std.exp()
		mean, action, log_prob = math.squash(mean, action, log_prob)		# 动作被放缩到[-1, 1]

		entropy_scale = scaled_log_prob / (log_prob + 1e-8)
		info = {
			"mean": mean,
			"log_std": log_std,
			"action_prob": torch.ones_like(log_prob),  # 或者 mean.new_ones(())
			"entropy": -log_prob,
			"scaled_entropy": -log_prob * entropy_scale,
		}

		return action, info

	def Q(self, z, a, task, return_type='min', target=False, detach=False):
		assert return_type in {'min', 'avg', 'all'}

		# ---- 强制二维 ----
		# z: [B,Z]
		if z.dim() == 3:
			assert z.size(0) == 1, f"Q expects z [B,Z], got {z.shape}"
			z = z[0]
		assert z.dim() == 2

		# a: [B,A]
		if a.dim() == 3:
			assert a.size(0) == 1, f"Q expects a [B,A], got {a.shape}"
			a = a[0]
		assert a.dim() == 2

		if self.cfg.multitask:
			z = self.task_emb(z, task)

		x = torch.cat([z, a], dim=-1)  # [B, Z+A]

		qnet = self._target_Qs if target else self._Qs
		out = qnet(x)  # [num_q, B, ...] or [B,...] 视实现而定
		if detach:
			out = out.detach()

		if return_type == 'all':
			return out

		qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
		Q = math.two_hot_inv(out[qidx], self.cfg)  # [2,B]
		if return_type == 'min':
			return Q.min(0).values  # [B]
		return Q.mean(0)  # [B]



