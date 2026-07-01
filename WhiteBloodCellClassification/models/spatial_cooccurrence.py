import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import MLPLayer, CoLLayer


class SCFEModule(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super().__init__()
        # MLP giữ NGUYÊN số kênh in_channels để có thể cộng residual với fi (Eq.2)
        self.mlp = MLPLayer(in_channels, hidden_dim, in_channels)
        self.co_occurrence = CoLLayer()
        # Linear chiếu từ in_channels (sau GAP) xuống out_channels (chiều query)
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        fi = x                                  # f_i gốc, in_channels
        I = self.mlp(fi)                        # I_p = MLP(f_i), giữ nguyên in_channels
        col_out = self.co_occurrence(I, fi)      # L tính từ I, giá trị tổng hợp lấy từ fi (Eq.1)
        y = fi + col_out                         # residual: f_i + CoL(MLP(f_i))  — đúng Eq.2
        y = F.adaptive_avg_pool2d(y, (1, 1))
        y = y.flatten(2).permute(0, 2, 1)        # (B, 1, in_channels)
        query = self.linear(y)                   # chiếu xuống (B, 1, out_channels) — q_i
        return query
# class SCFEModule(nn.Module): 
#     def __init__(self, in_channels, hidden_dim, out_channels): 
#         super().__init__() 
#         self.proj = nn.Conv2d(in_channels, out_channels, 1) 
#         self.mlp = MLPLayer(in_channels, hidden_dim, out_channels) 
#         self.co_occurrence = CoLLayer() 
#         self.linear = nn.Linear(out_channels, out_channels) 
        
#     def forward(self, x): 
#         fi = x
#         mlp_out = self.mlp(fi) 
#         col_out = self.co_occurrence(mlp_out) 
#         y = self.proj(fi) + col_out 
#         # BxCx1x1 
#         y = F.adaptive_avg_pool2d(y, (1,1)) 
#         y = y.flatten(2).permute(0,2,1) 
#         query = self.linear(y) 
#         # Bx1xC 
#         return query
