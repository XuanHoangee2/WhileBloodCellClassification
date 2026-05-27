import torch 
import torch.nn as nn
from models.classifier import NuclearCytoplasmicClassifier
from DomainAdaptation.DA_module import DomainAdaptationModule
from models.PixelEncoder import PixelEncoder
from models.PixelDecoder import PixelDecoder
from models.spatial_cooccurrence import SCFEModule
from models.segmentationHead import SegmentationHead
from models.TransformerDecoder import TransformerDecoder

class TaskModule(nn.Module):
    def __init__(self,pretrained = True,num_classes = 8):
        super(). __init__()
        self.pixel_encoder = PixelEncoder(pretrained=pretrained)
        self.pixel_decoder = PixelDecoder()
        self.scfe = SCFEModule(2048, 512, 256)
        self.segmentation_head = SegmentationHead()
        self.transformer_decoder = TransformerDecoder()
        self.classifier = NuclearCytoplasmicClassifier(num_classes=num_classes)
    
    def forward(self,x):
        feature = self.pixel_encoder(x)
        pixel_feature = self.pixel_decoder(feature)[0]
        qi = self.scfe(feature[-1])
        Qi = self.transformer_decoder(pixel_feature, qi)
        masks = self.segmentation_head(Qi, pixel_feature)
        logits = self.classifier(feature[-1], masks)

        return logits