import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
from config.config_loader import load_config

config = load_config()
img_height = config["Domain_Adaptation_training"]["IMAGE_HEIGHT"]
img_width = config["Domain_Adaptation_training"]["IMAGE_WIDTH"]

transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor()
])

class JSTCDataset(Dataset):
    def __init__(self,root_dir,mask_dir,transform = transform, indices=None):
        self.mask_dir = os.path.join(root_dir, mask_dir)
        self.transforms = transform
        self.indices = indices
        # self.binary_dir = os.path.join(root_dir, binary_dir)
        self.images = sorted([f for f in os.listdir(self.mask_dir) if f.endswith('.bmp')])
        self.mask = sorted([f for f in os.listdir(self.mask_dir) if f.endswith('.png')])
        # self.binary = sorted([f for f in os.listdir(self.binary_dir) if f.endswith('.png')])
        assert len(self.images) == len(self.mask), "Số lượng image và mask không khớp!"

    
    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self.images)
    def __getitem__(self, idx):
        if self.indices is not None:
            idx = self.indices[idx]
        img_name = self.images[idx]
        mask_name = self.mask[idx]
        # binary_name = self.binary[idx]
        img_path = os.path.join(self.mask_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        # binary_path = os.path.join(self.binary_dir, binary_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self.transforms(image)
        mask = cv2.imread(mask_path,0)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        label_mask = np.zeros_like(mask)
        label_mask[mask == 255] = 2
        label_mask[mask == 128] = 1
        label_mask[mask == 0] = 0
        label_mask = torch.tensor(label_mask).long()
        # binary = cv2.imread(binary_path,0)
        # binary = cv2.resize(binary, (256, 256), interpolation=cv2.INTER_NEAREST)
        # label_binary = np.zeros_like(binary)
        # label_binary[binary == 255] = 1
        # label_binary[binary == 0] = 0
        # label_binary = torch.tensor(label_binary).long()
        return {
            "image": image,
            "mask": label_mask,
        }