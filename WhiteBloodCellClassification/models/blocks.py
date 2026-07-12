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
    def __init__(self, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.spatial_weight = nn.Parameter(torch.randn(kernel_size * kernel_size))

    def forward(self, I, f):
        """
        I: đặc trưng đã qua MLP, dùng để tính độ đồng thời L(Ip, Iq)  (Eq.1)
        f: đặc trưng GỐC (chưa qua MLP), dùng làm f_q trong phép tổng trọng số (Eq.1)
        """
        B, C, H, W = I.shape
        _, Cf, _, _ = f.shape

        # Tính L([Ip],[Iq]) từ I
        I_patches = F.unfold(I, kernel_size=self.kernel_size, padding=self.padding)
        I_patches = I_patches.view(B, C, self.kernel_size * self.kernel_size, H, W)
        I_center = I.unsqueeze(2)
        similarity = F.cosine_similarity(I_center, I_patches, dim=1)
        likelihood = (similarity + 1.0) / 2.0          # L(Ip, Iq)

        w = self.spatial_weight.view(1, -1, 1, 1)
        weight = likelihood * w                          # w_q * L
        weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-6)

        # Tổng hợp giá trị từ f (đặc trưng GỐC) — đúng theo Eq.1: w_q * L * f_q
        f_patches = F.unfold(f, kernel_size=self.kernel_size, padding=self.padding)
        f_patches = f_patches.view(B, Cf, self.kernel_size * self.kernel_size, H, W)

        out = (f_patches * weight.unsqueeze(1)).sum(dim=2)
        return out
# class CoLLayer(nn.Module):
#     def __init__(self, kernel_size = 5):
#         super().__init__() 
#         self.kernel_size = kernel_size 
#         self.padding = kernel_size // 2 
#         self.spatial_weight = nn.Parameter(torch.randn(kernel_size*kernel_size)) 
#     def forward(self, x): 
#         B,C,H,W = x.shape 
#         # Bx(Cxkernel_size*kernel_size)x(HxW) 
#         patches = F.unfold(x,kernel_size=self.kernel_size, padding=self.padding) 
#         patches = patches.view(B,C,self.kernel_size*self.kernel_size,H,W) 
#         center = x.unsqueeze(2) 
#         similarity = F.cosine_similarity(center,patches,dim=1) 
#         likelihood = (similarity + 1.0) / 2.0 
#         w = self.spatial_weight.view(1,-1,1,1) 
#         # Bxkernel_size*kernel_sizex(HxW) 
#         weight = likelihood * w 
#         weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-6) 
#         # BxCx(HxW) 
#         out = (patches*weight.unsqueeze(1)).sum(dim=2) 
#         return out
