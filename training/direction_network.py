# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

# -----------------------------
# 工具：层索引嵌入 + 归一化
# -----------------------------
class LayerPosEnc(nn.Module):
    def __init__(self, L: int, d_pos: int):
        super().__init__()
        self.emb = nn.Embedding(L, d_pos)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, L: int, B: int) -> torch.Tensor:
        # [L, d_pos] -> [B, L, d_pos]
        idx = torch.arange(L, device=self.emb.weight.device)
        pos = self.emb(idx)[None].expand(B, -1, -1)
        return pos  # [B, L, d_pos]


def norm_tokens(x: torch.Tensor) -> torch.Tensor:
    # x: [B, L, d]  (LayerNorm per token)
    return F.layer_norm(x, x.shape[-1:])

# -----------------------------
# 适配器（轻量）：各空间 -> tokens [B, L, d]
# -----------------------------
class AdapterW(nn.Module):
    """ w ∈ R^{B,512} -> tokens [B,L,d] """
    def __init__(self, L: int, d: int, d_pos: int = 32, in_dim: int = 512):
        super().__init__()
        self.L, self.d = L, d
        self.proj = nn.Linear(in_dim + d_pos, d)
        self.pos = LayerPosEnc(L, d_pos)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        B = w.size(0)
        pos = self.pos(self.L, B)                      # [B, L, d_pos]
        w_rep = w[:, None, :].expand(B, self.L, -1)    # [B, L, 512]
        x = torch.cat([w_rep, pos], dim=-1)            # [B, L, 512+d_pos]
        tokens = self.proj(x)                          # [B, L, d]
        return norm_tokens(tokens)


class AdapterWPlus(nn.Module):
    """ w+ ∈ R^{B,L,512} -> tokens [B,L,d] (共享线性 + 层位置) """
    def __init__(self, L: int, d: int, d_pos: int = 32, in_dim: int = 512):
        super().__init__()
        self.L, self.d = L, d
        self.proj = nn.Linear(in_dim + d_pos, d)
        self.pos = LayerPosEnc(L, d_pos)

    def forward(self, w_plus: torch.Tensor) -> torch.Tensor:
        B, L, D = w_plus.shape
        assert L == self.L, f"Expected L={self.L}, got {L}"
        pos = self.pos(self.L, B)                      # [B, L, d_pos]
        x = torch.cat([w_plus, pos], dim=-1)           # [B, L, 512+d_pos]
        tokens = self.proj(x)                          # [B, L, d]
        return norm_tokens(tokens)


class AdapterS(nn.Module):
    """ s 空间：list of per-layer [B, d_l] -> tokens [B,L,d] (逐层轻量线性) """
    def __init__(self, dims_per_layer: List[int], d: int, d_pos: int = 32):
        super().__init__()
        self.L = len(dims_per_layer)
        self.projs = nn.ModuleList([nn.Linear(dl + d_pos, d) for dl in dims_per_layer])
        self.pos = LayerPosEnc(self.L, d_pos)

    def forward(self, s_list: List[torch.Tensor]) -> torch.Tensor:
        B = s_list[0].size(0)
        pos = self.pos(self.L, B)  # [B, L, d_pos]
        tokens = []
        for l, s_l in enumerate(s_list):
            x = torch.cat([s_l, pos[:, l, :]], dim=-1)     # [B, d_l + d_pos]
            tokens.append(self.projs[l](x))                # [B, d]
        tokens = torch.stack(tokens, dim=1)                # [B, L, d]
        return norm_tokens(tokens)


class AdapterTriplane(nn.Module):
    """
    triplane ∈ [B, 3, C, H, W] -> tokens [B, L, d]
    轻量策略：对每平面 GAP 得 [B,3,C] -> 线性到 base_tokens [B,3,d]
            再通过一个固定 mixing 矩阵 A 将 3 个 base token 混到 L 个 tokens
    """
    def __init__(self, L: int, d: int, C: int, d_pos: int = 32):
        super().__init__()
        self.L, self.d = L, d
        self.plane_proj = nn.Linear(C, d)             # per-plane to token
        self.mix = nn.Parameter(torch.randn(3, L) * 0.02)  # plane->L mixing
        self.pos = LayerPosEnc(L, d_pos)
        self.post = nn.Linear(d + d_pos, d)

    def forward(self, planes: torch.Tensor) -> torch.Tensor:
        # planes: [B,3,C,H,W]
        B, P, C, H, W = planes.shape
        assert P == 3
        gap = planes.mean(dim=(-1, -2))               # [B,3,C]
        base_tok = self.plane_proj(gap)               # [B,3,d]
        # mix到 L 个 tokens
        mix = F.softmax(self.mix, dim=-1)             # [3,L]
        tokens = torch.einsum('bpd,pl->bld', base_tok, mix)  # [B, L, d]
        pos = self.pos(self.L, B)
        tokens = self.post(torch.cat([tokens, pos], dim=-1)) # [B, L, d]
        return norm_tokens(tokens)

# -----------------------------
# 共享核心（完全共享、同一套参数）
# 你可改为 TransformerEncoder；这里给出 MLP-Transformer 混合示例
# -----------------------------
class SharedDirectionNet(nn.Module):
    def __init__(self, d: int, n_layers: int = 4, nhead: int = 4, ff: int = 4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=ff*d, batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d)  # 保持形状不变，残差 tokens
        )

    def forward(self, tokens: torch.Tensor,
                cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        # tokens: [B,L,d]
        # cond:   [B,dc] -> 简单注入：加到 tokens 的第一个 token 或全部 token
        if cond is not None:
            # 将 cond 投到 d 并加到所有 tokens 上（FiLM/加性调制的极简版）
            proj = F.layer_norm(cond, cond.shape[-1:])  # [B,dc]
            proj = F.pad(proj, (0, tokens.size(-1)-proj.size(-1))) if proj.size(-1) < tokens.size(-1) else proj[:, :tokens.size(-1)]
            tokens = tokens + proj[:, None, :]

        h = self.enc(tokens)                # [B,L,d]
        delta_tokens = self.out(h)          # [B,L,d]
        return delta_tokens

# -----------------------------
# 解码器（轻量）：tokens [B,L,d] -> Δ_res (原空间形状)
# -----------------------------
class DecoderW(nn.Module):
    """ tokens -> Δ_res_w [B,512] """
    def __init__(self, L: int, d: int, out_dim: int = 512):
        super().__init__()
        self.pool = nn.Linear(d, out_dim)  # 先线性投到 512，再平均
        self.alpha = nn.Parameter(torch.ones(L) / L)  # 学习型平均（轻量）

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # [B,L,d] -> [B,512]
        B, L, d = tokens.shape
        weights = F.softmax(self.alpha, dim=0)        # [L]
        pooled = (tokens * weights[None, :, None]).sum(dim=1)  # [B,d]
        out = self.pool(pooled)                       # [B,512]
        return out


class DecoderWPlus(nn.Module):
    """ tokens -> Δ_res_w+ [B,L,512] """
    def __init__(self, L: int, d: int, out_dim: int = 512):
        super().__init__()
        self.head = nn.Linear(d, out_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(tokens)  # [B,L,512]


class DecoderS(nn.Module):
    """ tokens -> Δ_res_s: list of per-layer residuals """
    def __init__(self, dims_per_layer: List[int], d: int):
        super().__init__()
        self.L = len(dims_per_layer)
        self.heads = nn.ModuleList([nn.Linear(d, dl) for dl in dims_per_layer])

    def forward(self, tokens: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        for l in range(self.L):
            outs.append(self.heads[l](tokens[:, l, :]))  # [B, d_l]
        return outs


class DecoderTriplane(nn.Module):
    """
    tokens -> Δ_res_planes [B,3,C,H,W]（仅输出 per-channel 偏置并广播到 H,W）
    更稳：不要预测密度的复杂卷积，只做颜色/辐射分量的通道偏置（你也可拆分密度/颜色）
    """
    def __init__(self, L: int, d: int, C: int, H: int, W: int):
        super().__init__()
        self.mix = nn.Parameter(torch.randn(L, 3) * 0.02)  # tokens->3 base
        self.to_ch = nn.Linear(d, C)
        self.H, self.W = H, W

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # [B,L,d] -> [B,3,C,H,W]
        B, L, d = tokens.shape
        mix = F.softmax(self.mix, dim=0)                   # [L,3]
        base = torch.einsum('bld,lp->bpd', tokens, mix)    # [B,3,d]
        ch = self.to_ch(base)                              # [B,3,C]
        out = ch[..., None, None].expand(B, 3, ch.size(-1), self.H, self.W)
        return out

# -----------------------------
# 统一封装：某空间 -> Δ_res（与原空间同形状）
# -----------------------------
class UnifiedDirectionEditor(nn.Module):
    """
    使用轻量适配器 + 共享核心 + 轻量解码器，
    在不同空间上公平地产生 Δ_res。
    """
    def __init__(self,
                 space: str,
                 L: int,
                 d: int,
                 dims_per_layer: Optional[List[int]] = None,
                 tri_spec: Optional[Tuple[int,int,int]] = None,  # (C,H,W)
                 shared_core: Optional[nn.Module] = None):
        super().__init__()
        self.space = space.lower()
        self.L, self.d = L, d

        # 适配器
        if self.space == "w":
            self.adapter = AdapterW(L=L, d=d)
            self.decoder = DecoderW(L=L, d=d)
        elif self.space == "w+":
            self.adapter = AdapterWPlus(L=L, d=d)
            self.decoder = DecoderWPlus(L=L, d=d)
        elif self.space == "s":
            assert dims_per_layer is not None
            self.adapter = AdapterS(dims_per_layer=dims_per_layer, d=d)
            self.decoder = DecoderS(dims_per_layer=dims_per_layer, d=d)
        elif self.space == "triplane":
            assert tri_spec is not None
            C, H, W = tri_spec
            self.adapter = AdapterTriplane(L=L, d=d, C=C)
            self.decoder = DecoderTriplane(L=L, d=d, C=C, H=H, W=W)
        else:
            raise ValueError("space must be one of {w, w+, s, triplane}")

        # 共享核心（可外部传入以“多空间共享”，不传则自己建一份）
        self.core = shared_core if shared_core is not None else SharedDirectionNet(d=d)

        # 条件编码（最简合并：把标量/小向量拼接后线性到 d 再注入）
        self.cond_enc = nn.Sequential(
            nn.Linear(64, d), nn.GELU(), nn.Linear(d, d)
        )

    def encode_cond(self,
                    alpha: torch.Tensor,
                    alpha_star: torch.Tensor,
                    t_emb: torch.Tensor,
                    c_emb: torch.Tensor) -> torch.Tensor:
        """
        alpha, alpha_star: [B,1]
        t_emb, c_emb:     [B, k_t], [B, k_c]  (已做过PE/embedding)
        """
        B = alpha.size(0)
        # 把各种标量/向量统一到 64 维（这里简单：拼接后 pad/crop 到 64）
        x = torch.cat([alpha, alpha_star, t_emb, c_emb], dim=-1)
        if x.size(-1) < 64:
            x = F.pad(x, (0, 64 - x.size(-1)))
        elif x.size(-1) > 64:
            x = x[:, :64]
        return self.cond_enc(x)  # [B,d]

    def forward(self,
                u_space,
                alpha: torch.Tensor,
                alpha_star: torch.Tensor,
                t_emb: torch.Tensor,
                c_emb: torch.Tensor):
        """
        返回 Δ_res（与原空间形状一致），以及中间 tokens，便于对齐对比/可视化
        """
        tokens = self.adapter(u_space)  # [B,L,d]
        cond = self.encode_cond(alpha, alpha_star, t_emb, c_emb)  # [B,d]
        delta_tokens = self.core(tokens, cond=cond)               # [B,L,d]
        delta_res = self.decoder(delta_tokens)                    # shape matches space
        return delta_res, tokens, delta_tokens

# -----------------------------
# 使用示例
# -----------------------------
if __name__ == "__main__":
    B = 2
    L = 14       # 256 分辨率常见 14 层
    d = 128

    # 共享核心：跨空间共用这一个
    shared_core = SharedDirectionNet(d=d, n_layers=4, nhead=4, ff=4)

    # 1) s 空间
    dims_s = [512,512,512,512,256,256,128,128,64,64,32,32,16,16]
    editor_s = UnifiedDirectionEditor(space="s", L=L, d=d, dims_per_layer=dims_s, shared_core=shared_core)

    s_list = [torch.randn(B, dl) for dl in dims_s]
    alpha = torch.rand(B,1); alpha_star = torch.rand(B,1)
    t_emb = torch.randn(B,8); c_emb = torch.randn(B,8)
    d_res_s, tok_s, dtok_s = editor_s(s_list, alpha, alpha_star, t_emb, c_emb)
    # d_res_s: list of 14 tensors, shapes match dims_s

    # 2) w+ 空间
    editor_wplus = UnifiedDirectionEditor(space="w+", L=L, d=d, shared_core=shared_core)
    w_plus = torch.randn(B, L, 512)
    d_res_wplus, _, _ = editor_wplus(w_plus, alpha, alpha_star, t_emb, c_emb)  # [B,L,512]

    # 3) w 空间
    editor_w = UnifiedDirectionEditor(space="w", L=L, d=d, shared_core=shared_core)
    w = torch.randn(B, 512)
    d_res_w, _, _ = editor_w(w, alpha, alpha_star, t_emb, c_emb)               # [B,512]

    # 4) triplane 空间
    C, H, W = 32, 64, 64
    editor_tri = UnifiedDirectionEditor(space="triplane", L=L, d=d, tri_spec=(C,H,W), shared_core=shared_core)
    planes = torch.randn(B, 3, C, H, W)
    d_res_tri, _, _ = editor_tri(planes, alpha, alpha_star, t_emb, c_emb)      # [B,3,C,H,W]

    print("OK: s/w+/w/triplane Δ_res shapes ready.")
