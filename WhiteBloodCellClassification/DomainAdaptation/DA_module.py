import torch
import torch.nn as nn
from models.PixelEncoder import PixelEncoder
from models.spatial_cooccurrence import SCFEModule
from models.PixelDecoder import PixelDecoder
from models.segmentationHead import SegmentationHead
from models.TransformerDecoder import TransformerDecoder

class DomainAdaptationModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.pixel_encoder = PixelEncoder()
        self.pixel_decoder = PixelDecoder()
        self.scfe = SCFEModule(2048, 512, 256)
        self.segmentation_head = SegmentationHead()
        self.transformer_decoder = TransformerDecoder()
        
    def forward(self, x):
        feature = self.pixel_encoder(x)
        pixel_feature = self.pixel_decoder(feature)[0]
        qi = self.scfe(feature[-1])
        Qi = self.transformer_decoder(pixel_feature, qi)
        masks = self.segmentation_head(Qi, pixel_feature)

        return masks, feature, Qi