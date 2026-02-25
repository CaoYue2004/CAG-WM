import torch
import torch.nn as nn

class BiLSTMSeqClassifier(nn.Module):
    def __init__(self,
                 input_dim: int = 3,     # (depth, density, pos) 等
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 bidirectional: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        out_dim = hidden_size * (2 if bidirectional else 1)

        # 可以先做一层投影再分类（效果通常更稳）
        self.proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.seq_head = nn.Linear(out_dim, 1)  # 序列级：每条序列一个 logit

    def forward(self, x, lengths):
        """
        x: [B, T, C], lengths: [B]
        返回：
          seq_logits: [B] 整条序列的 logit
        """
        # pack -> LSTM -> unpack
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # [B, T, H*D]

        # 基于 lengths 构造 mask（True=有效步）
        B, T, _ = out.shape
        device = out.device
        mask = (torch.arange(T, device=device)[None, :] < lengths[:, None])  # [B,T]

        # 有效步平均池化（也可换成 max/attention）
        out = self.proj(out)                     # [B,T,D]
        out = out * mask.unsqueeze(-1)           # [B,T,D]
        pooled = out.sum(dim=1) / lengths.unsqueeze(1)  # [B,D]

        seq_logits = self.seq_head(pooled).squeeze(-1)  # [B]
        return seq_logits
