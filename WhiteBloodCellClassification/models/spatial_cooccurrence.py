import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import MLPLayer, CoLLayer

class SCFEModule(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super().__init__()
        self.mlp = MLPLayer(in_channels, hidden_dim, out_channels)
        self.co_occurrence = CoLLayer()
        self.linear = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = self.mlp(x)
        y = self.co_occurrence(x)
        y = y + x
        y = y.mean(dim=[2,3])
        query = self.linear(y)
        return query
        