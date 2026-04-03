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
segmen = SegmentationHead()

TransformerDecoder = TransformerDecoder()

img_path = r"D:\work\WBC_Segmentation\WhileBloodCellClassification\data/RawData\Dataset 1/013.bmp"
image = Image.open(img_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
img_resized = transform(image).unsqueeze(0)  

features = encoder(img_resized)
pixel_features = decoder(features)[0]
qi = SCFE(features[-1])
Qi= TransformerDecoder(pixel_features, qi)
masks = segmen(Qi, pixel_features)

pred = masks.squeeze(0)
mask = pred.argmax(dim=0)

import numpy as np

mask_np = mask.cpu().numpy()

# scale về 0–255
mask_vis = (mask_np * (255 // 2)).astype(np.uint8)

plt.imshow(mask_vis, cmap='gray')
plt.show()




# import matplotlib.pyplot as plt

# img = masks[0].detach().cpu()  # shape: (3, 256, 256)
# img = img.permute(1, 2, 0)     # -> (256, 256, 3)

# plt.figure(figsize=(10,10))
# plt.imshow(img.numpy())
# plt.axis('off')
# plt.show()




