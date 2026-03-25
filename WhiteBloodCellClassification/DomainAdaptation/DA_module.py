import torch
import torch.nn as nn
from ..models.PixelEncoder import PixelEncoder
from ..models.spatial_cooccurrence import SCFEModule
from ..models.PixelDecoder import PixelDecoder
from ..models.segmentationHead import SegmentationHead
from ..models.TransformerDecoder import TransformerDecoder

# pixel_encoder = PixelEncoder()
# pixel_decoder = PixelDecoder()
# SCFE = SCFEModule(2048, 512, 256)
# segmentation_head = SegmentationHead(256, 256, 2048)
# transformer_decoder = TransformerDecoder()
class DomainAdaptationModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.pixel_encoder = PixelEncoder()
        self.pixel_decoder = PixelDecoder()
        self.scfe = SCFEModule(2048, 512, 256)
        self.segmentation_head = SegmentationHead(256, 256, 2048)
        self.transformer_decoder = TransformerDecoder()
        
    def forward(self, x):
        feature = self.pixel_encoder(x)
        pixel_feature = self.pixel_decoder(feature)
        qi = self.scfe(feature[-1])
        Qi = self.transformer_decoder(pixel_feature, qi)
        class_logit, masks = self.segmentation_head(Qi, pixel_feature, feature[-1])

        return masks, feature[-1], Qi