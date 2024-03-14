import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, seq_len, pred_len, num_layer=1, enc_in=1, individual=False):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_layer = num_layer

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.individual = individual
        self.channels = enc_in
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(
                    nn.Sequential(
                        *(nn.Linear(self.seq_len, self.seq_len) for _ in range(num_layer - 1)),
                        nn.Linear(self.seq_len, self.pred_len)
                    )
                )
        else:
            self.Linear = nn.Sequential(
                *(nn.Linear(self.seq_len, self.seq_len) for _ in range(num_layer - 1)),
                nn.Linear(self.seq_len, self.pred_len)
            )

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x  # [Batch, Output length, Channel]