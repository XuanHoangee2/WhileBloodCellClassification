import torch
import torch.nn as nn
import torch.nn.functional as F

class CoOccurrenceModule(nn.Module):
    def __init__(self, kernel_size = 5):
        super().__init__()

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.spatial_weight = nn.Parameter(torch.randn(kernel_size*kernel_size))
    
    def forward(self, x):
        B,C,H,W = x.shape
        patches = F.unfold(x,kernel_size=self.kernel_size, padding=self.padding)
        patches = patches.view(B,C,self.kernel_size*self.kernel_size,H,W)
        center = x.unsqueeze(2)
        similarity = F.cosine_similarity(center,patches,dim=1)
        likelihood = (similarity + 1.0) / 2.0
        w = self.spatial_weight.view(1,-1,1,1)
        weight = likelihood * w
        out = (patches*weight.unsqueeze(1)).sum(dim=2)
        return out
