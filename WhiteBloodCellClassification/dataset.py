import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
from config.config_loader import load_config

try:
    from augmentation import get_training_augmentation, get_validation_augmentation
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

config = load_config()
img_height = config["Domain_Adaptation_training"]["IMAGE_HEIGHT"]
img_width = config["Domain_Adaptation_training"]["IMAGE_WIDTH"]

transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor()
])

class JSTCDataset(Dataset):
    def __init__(self, root_dir, mask_dir, transform=None, indices=None,
                 use_augmentation=False, is_training=True, image_size=(256, 256)):
        self.mask_dir = os.path.join(root_dir, mask_dir)
        self.transforms = transform
        self.indices = indices
        self.use_augmentation = use_augmentation and ALBUMENTATIONS_AVAILABLE
        self.is_training = is_training
        self.image_size = image_size

        # Initialize albumentations transforms if requested
        if self.use_augmentation:
            if is_training:
                self.aug_transform = get_training_augmentation(image_size)
            else:
                self.aug_transform = get_validation_augmentation(image_size)
        else:
            self.aug_transform = None

        self.images = sorted([f for f in os.listdir(self.mask_dir) if f.endswith('.bmp')])
        self.mask = sorted([f for f in os.listdir(self.mask_dir) if f.endswith('.png')])
        assert len(self.images) == len(self.mask), "Number of images and masks do not match!"


    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self.images)

    def __getitem__(self, idx):
        if self.indices is not None:
            idx = self.indices[idx]

        img_name = self.images[idx]
        mask_name = self.mask[idx]

        img_path = os.path.join(self.mask_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Apply label mapping
        label_mask = np.zeros_like(mask)
        label_mask[mask == 255] = 2  # nucleus
        label_mask[mask == 128] = 1  # cytoplasm
        label_mask[mask == 0] = 0    # background

        # Apply augmentations
        if self.aug_transform is not None:
            augmented = self.aug_transform(image=image, mask=label_mask)
            image = augmented['image']
            label_mask = augmented['mask']
            # Ensure mask is long tensor after albumentations
            label_mask = torch.tensor(label_mask).long()
        else:
            # Legacy transform pipeline
            image = Image.fromarray(image)
            if self.transforms:
                image = self.transforms(image)
            else:
                image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

            label_mask = cv2.resize(label_mask, (self.image_size[1], self.image_size[0]),
                                   interpolation=cv2.INTER_NEAREST)
            label_mask = torch.tensor(label_mask).long()

        return {
            "image": image,
            "mask": label_mask,
            "image_name": img_name
        }