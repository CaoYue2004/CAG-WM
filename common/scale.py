import torch


class RunningScale(torch.nn.Module):
	"""Running trimmed scale estimator."""

	def __init__(self, cfg, device=None):
		super().__init__()
		self.cfg = cfg

		# 不要硬编码 cuda:0；让外部 .to(device) 决定，或者用 device 参数
		if device is None:
			# 如果你一定要默认上 GPU，可以用下面这一句；否则默认 cpu 更稳
			device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

		# 用 register_buffer 替代 torch.nn.Buffer（兼容 PyTorch 1.x/2.x）
		self.register_buffer(
			"value",
			torch.ones(1, dtype=torch.float32, device=device)
		)
		self.register_buffer(
			"_percentiles",
			torch.tensor([5, 95], dtype=torch.float32, device=device)
		)

	# ✅ 不要重写 state_dict / load_state_dict
	# PyTorch Module 自带的 state_dict 会自动包含 buffer：
	#   {"value": ..., "_percentiles": ...}
	# 也会自动处理 device/copy_


	def _positions(self, x_shape):
		positions = self._percentiles * (x_shape - 1) / 100
		floored = torch.floor(positions)
		ceiled = floored + 1
		ceiled = torch.where(ceiled > x_shape - 1, x_shape - 1, ceiled)
		weight_ceiled = positions - floored
		weight_floored = 1.0 - weight_ceiled
		return floored.long(), ceiled.long(), weight_floored.unsqueeze(1), weight_ceiled.unsqueeze(1)

	def _percentile(self, x):
		x_dtype, x_shape = x.dtype, x.shape
		if x.ndim <= 1:
			x = x.flatten(0)  # [B] 或标量 -> [N]
		else:
			x = x.flatten(1, x.ndim - 1)
		in_sorted = torch.sort(x, dim=0).values
		floored, ceiled, weight_floored, weight_ceiled = self._positions(x.shape[0])
		d0 = in_sorted[floored] * weight_floored
		d1 = in_sorted[ceiled] * weight_ceiled
		return (d0 + d1).reshape(-1, *x_shape[1:]).to(x_dtype)

	def update(self, x):
		percentiles = self._percentile(x.detach())
		value = torch.clamp(percentiles[1] - percentiles[0], min=1.)
		# 这里用 data.lerp_ 沿用你的写法；也可以用 torch.no_grad() 包起来
		self.value.data.lerp_(value, self.cfg.tau)

	def forward(self, x, update=False):
		if update:
			self.update(x)
		return x / self.value

	def __repr__(self):
		return f"RunningScale(S: {self.value})"
