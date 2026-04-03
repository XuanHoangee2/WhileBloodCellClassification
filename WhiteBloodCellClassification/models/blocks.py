import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class MLPLayer(nn.Module): 
    def __init__(self, in_channels, hidden_dim, out_channels): 
        super().__init__()
        mid_dim = (in_channels + hidden_dim) // 2
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid_dim, kernel_size=1),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        return self.mlp(x)

class CoLLayer(nn.Module):
    def __init__(self, kernel_size = 5):
        super().__init__() 
        self.kernel_size = kernel_size 
        self.padding = kernel_size // 2 
        self.spatial_weight = nn.Parameter(torch.randn(kernel_size*kernel_size)) 
    def forward(self, x): 
        B,C,H,W = x.shape 
        # Bx(Cxkernel_size*kernel_size)x(HxW) 
        patches = F.unfold(x,kernel_size=self.kernel_size, padding=self.padding) 
        patches = patches.view(B,C,self.kernel_size*self.kernel_size,H,W) 
        center = x.unsqueeze(2) 
        similarity = F.cosine_similarity(center,patches,dim=1) 
        likelihood = (similarity + 1.0) / 2.0 
        w = self.spatial_weight.view(1,-1,1,1) 
        # Bxkernel_size*kernel_sizex(HxW) 
        weight = likelihood * w 
        weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-6) 
        # BxCx(HxW) 
        out = (patches*weight.unsqueeze(1)).sum(dim=2) 
        return out
