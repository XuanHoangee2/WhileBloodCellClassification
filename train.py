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
segmen = SegmentationHead(256, 256)

TransformerDecoder = TransformerDecoder()

img_path = "data/Dataset 1/001.bmp"
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
class_logit, masks = segmen(Qi, pixel_features)
print(masks.shape)

# num_queries = 100 
# qi_input = qi.unsqueeze(0).expand(num_queries, -1, -1)

# pixel_features = decoder(features)
# print(features[1].shape)
# print(ouput.shape)
# plt.figure(figsize=(10,10))

# for i in range(16):
#     plt.subplot(4,4,i+1)
#     # plt.imshow(ouput[0,i].detach().cpu().numpy(), cmap='gray')
#     plt.imshow(output[0,i].view(16,16).detach().cpu().numpy(), cmap='gray')
#     plt.axis("off")

# plt.show()

plt.figure(figsize=(10, 10))

# Chọn 16 queries đầu tiên để in
for i in range(16):
    plt.subplot(4, 4, i + 1)
    
    # 1. Lấy mask thứ i
    # 2. Dùng .sigmoid() để đưa về khoảng 0-1 cho dễ nhìn
    # 3. Chuyển sang numpy để vẽ
    mask_to_plot = masks[0, i].sigmoid().detach().cpu().numpy()
    
    plt.imshow(mask_to_plot, cmap='gray')
    plt.title(f"Query {i}", fontsize=8)
    plt.axis("off")

plt.tight_layout()
plt.show()