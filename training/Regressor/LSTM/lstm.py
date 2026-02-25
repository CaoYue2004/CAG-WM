# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : model_LSTM.py
# @Last Modify Time : 2018/07/19 22:35
# @Contact : bamtercelboo@{gmail.com, 163.com}

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn.init as init

"""
Neural Networks model : LSTM
"""

class LSTM(nn.Module):

    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args

        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers

        resnet_output_dim = args.resnet_out_dim  # 例如 2048
        self.lstm = nn.LSTM(resnet_output_dim, self.hidden_dim,
                            dropout=args.dropout, num_layers=self.num_layers)

        if args.init_weight:
            print("Initing LSTM weights .......")
            init.xavier_normal_(self.lstm.all_weights[0][0], gain=np.sqrt(args.init_weight_value))
            init.xavier_normal_(self.lstm.all_weights[0][1], gain=np.sqrt(args.init_weight_value))

        self.hidden2label = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        # x shape: [batch_size, seq_len, feature_dim] ➜ from ResNet
        # Permute for LSTM: [seq_len, batch_size, feature_dim]
        x = x.permute(1, 0, 2)

        lstm_out, _ = self.lstm(x)  # [seq_len, batch_size, hidden_dim]

        lstm_out = lstm_out.permute(1, 2, 0)  # [batch, hidden_dim, seq_len]

        lstm_out = F.tanh(lstm_out)
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        lstm_out = F.tanh(lstm_out)

        logit = self.hidden2label(lstm_out)
        return logit
