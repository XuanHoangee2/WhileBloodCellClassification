from WhiteBloodCellClassification.models.PixelEncoder import PixelEncoder
from WhiteBloodCellClassification.models.PixelDecoder import PixelDecoder
import torch

encoder = PixelEncoder()
decoder = PixelDecoder()

x = torch.randn(1, 3, 256, 256)
features = encoder(x)
pixel_features = decoder(features)
print(pixel_features.shape)