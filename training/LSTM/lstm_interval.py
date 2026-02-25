import torch
import torch.nn as nn

class BiLSTMIntervalTagger(nn.Module):
    def __init__(self,
                 input_dim=3,
                 hidden_size=128,
                 num_layers=2,
                 dropout=0.2,
                 bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers>1 else 0.0,
            bidirectional=bidirectional
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_dim, 3)  # 3类: O/B/I
        )

    def forward(self, x, lengths=None):
        """
        x: [B,T,C]
        lengths: [B] or None
        如果等长（或不给lengths），直接走无pack分支；否则走pack。
        """
        if (lengths is None) or (lengths.min() == lengths.max()):
            out, _ = self.lstm(x)                      # [B,T,H*D]
        else:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        logits = self.head(out)                        # [B,T,3]
        return logits
