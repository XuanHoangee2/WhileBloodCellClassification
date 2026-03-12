import torch
import torch.nn as nn
from .blocks import ConvBlock
import torch.nn.functional as F

class PixelDecoder(nn.Module):
    def __init__(self, out_channels = 256):
        super().__init__()

        self.lateral4 = nn.Conv2d(2048, out_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(1024, out_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(512, out_channels, kernel_size=1)
        self.lateral1 = nn.Conv2d(256, out_channels, kernel_size=1)

        self.conv4 = ConvBlock(out_channels, out_channels)
        self.conv3 = ConvBlock(out_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.conv1 = ConvBlock(out_channels, out_channels)
    
    def forward(self, features):
        c1,c2,c3,c4 = features
        p4 = self.conv4(self.lateral4(c4))
        p3 = self.lateral3(c3) + F.interpolate(p4,scale_factor=2, mode='bilinear', align_corners=False)
        p3 = self.conv3(p3)
        p2 = self.lateral2(c2) + F.interpolate(p3,scale_factor=2, mode='bilinear', align_corners=False)
        p2 = self.conv2(p2)
        p1 = self.lateral1(c1) + F.interpolate(p2,scale_factor=2, mode='bilinear', align_corners=False)
        p1 = self.conv1(p1)
        return p1
    
