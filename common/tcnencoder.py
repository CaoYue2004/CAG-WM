import torch
import torch.nn as nn
import torch.nn.functional as F

class CurveTCNEncoder(nn.Module):
    """
    Encode a fixed-length 3D point sequence:
        (B, L, 3) -> (B, 2*d_out)   (mean pool + max pool concat)
    """
    def __init__(self, d=128, d_out=128, k=3):
        super().__init__()
        pad = k // 2  # keep length
        self.in_proj = nn.Linear(3, d)
        self.conv1 = nn.Conv1d(d, d, kernel_size=k, padding=pad)
        self.conv2 = nn.Conv1d(d, d_out, kernel_size=k, padding=pad)
        self.act = nn.ReLU(inplace=True)

    def forward(self, pts):
        """
        pts:
          - (B, L, 3)            single step
          - (H, B, L, 3)         sequence batch (H = horizon or H+1)
        return:
          - (B, 2*d_out)
          - (H, B, 2*d_out)
        """
        assert pts.shape[-1] == 3, f"last dim must be 3, got {pts.shape}"

        seq_mode = (pts.dim() == 4)  # (H,B,L,3)
        if seq_mode:
            H, B, L, C = pts.shape
            pts = pts.reshape(H * B, L, C)  # -> (HB, L, 3)
        else:
            # ensure (B,L,3)
            if pts.dim() == 2:  # (L,3)
                pts = pts.unsqueeze(0)

        x = self.in_proj(pts)  # (HB, L, d) or (B, L, d)
        x = x.permute(0, 2, 1).contiguous()  # (HB, d, L)  ✅ Conv1d needs this

        x = self.act(self.conv1(x))  # (HB, d, L)
        x = self.act(self.conv2(x))  # (HB, d_out, L)

        x_mean = x.mean(dim=-1)  # (HB, d_out)
        x_max = x.max(dim=-1).values  # (HB, d_out)
        out = torch.cat([x_mean, x_max], dim=-1)  # (HB, 2*d_out)

        if seq_mode:
            out = out.view(H, B, -1)  # (H, B, 2*d_out)

        return out


class DictObsEncoderTCN(nn.Module):
    """
    obs dict:
      position: (B,20,3) or (20,3)
      route:    (B,32,3) or (32,3)
      target:   (B,3)    or (3,)
      rotation: (B,1,2) or (B,2) or (1,2) or (2,)
    output: (B, out_dim)
    """
    def __init__(self, d=128, d_out=128, out_dim=256):
        super().__init__()

        self.pos_enc = CurveTCNEncoder(d=d, d_out=d_out)
        self.route_enc = CurveTCNEncoder(d=d, d_out=d_out)

        self.small_enc = nn.Sequential(
            nn.Linear(5, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
        )

        fused_in = (2*d_out) + (2*d_out) + d  # pos + route + (target+rot)
        self.fuse = nn.Sequential(
            nn.Linear(fused_in, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    @staticmethod
    def _ensure_batch(x):
        # (L, C) -> (1, L, C) ; (C,) -> (1, C)
        if x is None:
            return None
        if x.dim() == 1:
            return x.unsqueeze(0)
        if x.dim() == 2:
            return x.unsqueeze(0)
        return x

    def forward(self, obs: dict):
        pos = torch.as_tensor(obs["position"])
        route = torch.as_tensor(obs["route"])
        target = torch.as_tensor(obs["target"])
        rot = torch.as_tensor(obs["rotation"])

        # detect sequence batch: target is (H,B,3) or rot is (H,B,1,2)
        seq_mode = (target.dim() == 3 and target.shape[-1] == 3 and pos.dim() == 4)
        if seq_mode:
            H, B = target.shape[0], target.shape[1]

            # (H,B,L,3) -> (HB,L,3)
            pos = pos.reshape(H * B, *pos.shape[2:])
            route = route.reshape(H * B, *route.shape[2:])

            # (H,B,3) -> (HB,3)
            target = target.reshape(H * B, *target.shape[2:])

            # (H,B,1,2) or (H,B,2) -> (HB,1,2)/(HB,2)
            rot = rot.reshape(H * B, *rot.shape[2:])

        # Ensure batch dims
        pos = self._ensure_batch(pos)       # (B,20,3)
        route = self._ensure_batch(route)   # (B,32,3)
        target = self._ensure_batch(target) # (B,3)

        # rotation could be (B,1,2) / (1,2) / (2,)
        rot = self._ensure_batch(rot)       # -> (B,1,2) or (B,2)
        if rot.dim() == 3:                  # (B,1,2) -> (B,2)
            rot = rot.view(rot.shape[0], -1)
        elif rot.dim() == 2 and rot.shape[1] != 2:
            rot = rot.view(rot.shape[0], -1)
        # print(f'pos={pos.shape}, route={route.shape}, target={target.shape}, rot={rot.shape}')
        # pos=torch.Size([1, 20, 3]), route=torch.Size([1, 32, 3]), target=torch.Size([1, 1, 3]), rot=torch.Size([1, 2])

        h_pos = self.pos_enc(pos)           # (B,2*d_out)
        h_route = self.route_enc(route)     # (B,2*d_out)
        # target0 = torch.Size([1, 1, 3]), rot0 = torch.Size([1, 2])
        # target0=torch.Size([3, 512, 3]), rot0=torch.Size([3, 512, 1, 2])
        # print(f'target0={target.shape}, rot0={rot.shape}')

        # target: (B,1,3) -> (B,3)
        if target.dim() == 3:
            if target.shape[0] == 1 and target.shape[-1] == 3:
                target = target.squeeze(0)  # (N,3)
            elif target.shape[1] == 1 and target.shape[-1] == 3:
                target = target.squeeze(1)  # (N,3)
            else:
                target = target.reshape(-1, 3)  # (...,3) -> (N,3)

        if rot.dim() > 2:
            rot = rot.view(rot.shape[0], -1)

        # print(f'target1={target.shape}, rot1={rot.shape}')
        small = torch.cat([target, rot], dim=-1)  # (B,5)
        # print(f'target={target.shape}, rot={rot.shape}, small={small.shape}')
        h_small = self.small_enc(small)     # (B,d)

        h = torch.cat([h_pos, h_route, h_small], dim=-1)
        out = self.fuse(h)
        if seq_mode:
            out = out.view(H, B, -1)
        return out


