import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 工具：把 [B,V,C,H,W] 攤平成 [B*V,C,H,W]，或原样返回 ---
def flatten_views(x):
    if x.dim() == 5:  # [B,V,C,H,W]
        B, V, C, H, W = x.shape
        return x.reshape(B*V, C, H, W), (B, V)
    return x, None

def restore_views(x, shape_BV):
    if shape_BV is None:
        return x
    B, V = shape_BV
    return x.reshape(B, V, *x.shape[1:])

# ---- 生成二维高斯二阶导数核（带尺度归一化） ----
def _gauss_2nd_kernels(sigma, device, dtype):
    # 核大小：保证奇数，覆盖 ~±3σ
    k = int(round(6*sigma)) | 1
    half = k // 2
    y, x = torch.meshgrid(
        torch.arange(-half, half+1, device=device, dtype=dtype),
        torch.arange(-half, half+1, device=device, dtype=dtype),
        indexing='ij'
    )
    sigma2 = sigma * sigma
    norm = 1.0 / (2.0 * math.pi * sigma2)
    g = norm * torch.exp(-(x**2 + y**2) / (2.0*sigma2))

    # 二阶导（LoG 家族公式）
    Gxx = (x**2 / sigma2 - 1.0) / sigma2 * g
    Gyy = (y**2 / sigma2 - 1.0) / sigma2 * g
    Gxy = (x * y / sigma2)     / sigma2 * g

    # Frangi 里常用尺度归一化：σ^2 * Lxx 等
    scale = sigma2
    Gxx = Gxx * scale
    Gyy = Gyy * scale
    Gxy = Gxy * scale
    # [1,1,k,k] 形状，便于 conv2d
    return (Gxx[None,None], Gxy[None,None], Gyy[None,None])

# ---- Frangi 血管度（多尺度最大化） ----
@torch.no_grad()
def frangi_vesselness(img, sigmas=(1.0, 2.0, 3.0), beta=0.5, c=15.0, black_ridges=True):
    """
    img: [N,C,H,W] (值建议 [0,1])；会自动转灰度
    返回: vesselness ∈ [0,1], 形状 [N,1,H,W]
    """
    N, C, H, W = img.shape
    device, dtype = img.device, img.dtype
    I = img.mean(dim=1, keepdim=True)  # 转灰度

    vessel_max = torch.zeros((N,1,H,W), device=device, dtype=dtype)

    for s in sigmas:
        Gxx, Gxy, Gyy = _gauss_2nd_kernels(s, device, dtype)
        Lxx = F.conv2d(I, Gxx, padding=Gxx.shape[-1]//2)
        Lxy = F.conv2d(I, Gxy, padding=Gxy.shape[-1]//2)
        Lyy = F.conv2d(I, Gyy, padding=Gyy.shape[-1]//2)

        # 2x2 Hessian 的特征值（闭式解）
        tmp = torch.sqrt((Lxx - Lyy)**2 + 4.0*(Lxy**2) + 1e-12)
        l1 = 0.5 * (Lxx + Lyy - tmp)
        l2 = 0.5 * (Lxx + Lyy + tmp)

        # 令 |l1| <= |l2|
        swap = (l1.abs() > l2.abs())
        l1_, l2_ = l1.clone(), l2.clone()
        l1[swap], l2[swap] = l2_[swap], l1_[swap]

        # 亮血管（bright ridges）用 l2<0；暗血管（black ridges）用 l2>0
        if black_ridges:
            mask = (l2 > 0)
        else:
            mask = (l2 < 0)

        Ra = (l1.abs() / (l2.abs() + 1e-12))                # 形状指数
        S  = torch.sqrt(l1**2 + l2**2 + 1e-12)              # 结构强度

        V = torch.exp(-(Ra**2) / (2*beta*beta)) * (1.0 - torch.exp(-(S**2) / (2*c*c)))
        V = V * mask  # 只在期望极性处保留

        vessel_max = torch.maximum(vessel_max, V)

    # 归一化到 [0,1]
    vessel_max = torch.clamp(vessel_max, 0.0, 1.0)
    return vessel_max

# ---- 用 Frangi 生成“背景”软掩膜（背景=1, 血管=0） ----
@torch.no_grad()
def estimate_bg_mask_frangi(img, sigmas=(1.0,2.0,3.0), beta=0.5, c=15.0, black_ridges=True):
    """
    img: [N,C,H,W], [0,1]
    返回 bg_mask ∈ [0,1]，背景大、血管小
    """
    vessel = frangi_vesselness(img, sigmas=sigmas, beta=beta, c=c, black_ridges=black_ridges)
    # 反相得到“背景”概率；可选再做轻度平滑
    bg = 1.0 - vessel
    return bg

# --- 三个 Loss 的整合 ---
class LossPack(nn.Module):
    def __init__(self, lambda_attr=1.0, lambda_bg=0.1, lambda_id=0.1):
        super().__init__()
        self.lambda_attr = lambda_attr
        self.lambda_bg   = lambda_bg
        self.lambda_id   = lambda_id
        self.mse = nn.MSELoss()

    def forward(self, *,
                I_orig,          # 原图像 [B,C,H,W] 或 [B,V,C,H,W]
                I_edit,          # 编辑后图像，同上
                R,               # 属性回归器（冻结或可微均可，一般冻结但要保梯度链）
                alpha_star,      # 目标属性 [B,1] 或 [B,V,1]（若多视角，按同视角复制即可）
                arcface,         # 你的 ArcFace 模型（一般冻结，但允许反传）
                mask=None        # 可选：血管/前景软掩膜，形状与 I_* 对齐（单通道）
                ):
        # 展平视角，统一计算
        I_orig_f, shp = flatten_views(I_orig)
        I_edit_f, _   = flatten_views(I_edit)

        # 归一化到 [0,1]（若你管线已是 [0,1] 可去掉）
        I_orig_f = torch.clamp(I_orig_f, 0, 1)
        I_edit_f = torch.clamp(I_edit_f, 0, 1)

        # -------- L_attr：属性可控性 --------
        # R 要对 I_edit 做预测；alpha_star 需要与视角维度对齐
        if alpha_star.dim() == 3:  # [B,V,1]
            alpha_star_f = alpha_star.reshape(-1, alpha_star.size(-1))  # [B*V,1]
        else:
            # [B,1] -> 若多视角，复制到 [B*V,1]
            if shp is not None:
                B, V = shp
                alpha_star_f = alpha_star.repeat_interleave(V, dim=0)
            else:
                alpha_star_f = alpha_star

        alpha_hat = R(I_edit_f)  # 形状应为 [N,1] 或 [N]；如不是请在外部对齐
        if alpha_hat.dim() == 1:
            alpha_hat = alpha_hat.unsqueeze(-1)

        L_attr = self.mse(alpha_hat, alpha_star_f)

        # -------- L_bg：背景稳定（无 mask 则用低梯度近似背景） --------
        if mask is None:
            bg_mask, _ = flatten_views(estimate_bg_mask_frangi(I_orig))
        else:
            bg_mask, _ = flatten_views(mask)  # 期望 [N,1,H,W] 或 [N,H,W]；若是后者请在外部加 channel 维
            if bg_mask.dim() == 3:
                bg_mask = bg_mask.unsqueeze(1)
        # L1：仅在背景区域惩罚变化
        # 把图像差取均值到单通道后再乘 mask（避免通道数不同）
        diff = (I_edit_f - I_orig_f).abs().mean(dim=1, keepdim=True)  # [N,1,H,W]
        L_bg = ( (bg_mask) * diff ).mean()

        # -------- L_id：ArcFace 身份一致性 --------
        # ArcFace 输入通常为归一化/对齐图；如需中心裁剪/灰度转 RGB，请在外部预处理
        z_o = F.normalize(arcface(I_orig_f), dim=1)
        z_e = F.normalize(arcface(I_edit_f), dim=1)
        cos_sim = (z_o * z_e).sum(dim=1)  # [N]
        L_id = 1.0 - cos_sim.mean()

        # -------- 总损失 --------
        loss = (self.lambda_attr * L_attr
                + self.lambda_bg * L_bg
                + self.lambda_id * L_id)

        # 返回逐项便于日志
        return {
            "loss": loss,
            "L_attr": L_attr.detach(),
            "L_bg": L_bg.detach(),
            "L_id": L_id.detach(),
        }
