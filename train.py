from WhiteBloodCellClassification.models.PixelEncoder import PixelEncoder
from WhiteBloodCellClassification.models.PixelDecoder import PixelDecoder
from WhiteBloodCellClassification.models.TransformerDecoder import TransformerDecoder
from WhiteBloodCellClassification.models.segmentationHead import SegmentationHead
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt 
from torchvision import transforms
from WhiteBloodCellClassification.models.spatial_cooccurrence import SCFEModule
from WhiteBloodCellClassification.models.blocks import MLPLayer

encoder = PixelEncoder()
decoder = PixelDecoder()
SCFE = SCFEModule(2048, 512, 256)
segmen = SegmentationHead(256, 256,2048)

TransformerDecoder = TransformerDecoder()

img_path = r"D:\work\WBC_Segmentation\WhileBloodCellClassification\data/RawData\Dataset 1/013.bmp"
image = Image.open(img_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
img_resized = transform(image).unsqueeze(0)  

features = encoder(img_resized)
pixel_features = decoder(features)
qi = SCFE(features[-1])
Qi= TransformerDecoder(pixel_features, qi)
class_logit, masks = segmen(Qi, pixel_features, features[-1])
print(class_logit.shape)
# import matplotlib.pyplot as plt

# img = masks[0].detach().cpu()  # shape: (3, 256, 256)
# img = img.permute(1, 2, 0)     # -> (256, 256, 3)

# plt.figure(figsize=(10,10))
# plt.imshow(img.numpy())
# plt.axis('off')
# plt.show()




