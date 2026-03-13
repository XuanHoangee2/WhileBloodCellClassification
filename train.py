from WhiteBloodCellClassification.models.PixelEncoder import PixelEncoder
from WhiteBloodCellClassification.models.PixelDecoder import PixelDecoder
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt 
from torchvision import transforms
from WhiteBloodCellClassification.models.spatial_cooccurrence import CoOccurrenceModule
from WhiteBloodCellClassification.models.blocks import MLPLayer

encoder = PixelEncoder()
decoder = PixelDecoder()
cooccurrence = CoOccurrenceModule()
MLP = MLPLayer(2048, 512, 256)

img_path = "data/Dataset 1/001.bmp"
image = Image.open(img_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
img_resized = transform(image).unsqueeze(0)  

features = encoder(img_resized)
I = MLP(features[-1])
Jp = cooccurrence(I)

# pixel_features = decoder(features)
# print(features[1].shape)
print(Jp.shape)
plt.figure(figsize=(10,10))

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(Jp[0,i].detach().cpu().numpy(), cmap='gray')
    plt.axis("off")

plt.show()