import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import MLPLayer, CoLLayer

class SCFEModule(nn.Module): 
    def __init__(self, in_channels, hidden_dim, out_channels): 
        super().__init__() 
        self.proj = nn.Conv2d(in_channels, out_channels, 1) 
        self.mlp = MLPLayer(in_channels, hidden_dim, out_channels) 
        self.co_occurrence = CoLLayer() 
        self.linear = nn.Linear(out_channels, out_channels) 
        
    def forward(self, x): 
        fi = x
        mlp_out = self.mlp(fi) 
        col_out = self.co_occurrence(mlp_out) 
        y = self.proj(fi) + col_out 
        # BxCx1x1 
        y = F.adaptive_avg_pool2d(y, (1,1)) 
        y = y.flatten(2).permute(0,2,1) 
        query = self.linear(y) 
        # Bx1xC 
        return query
