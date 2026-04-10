import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch


def get_training_augmentation(image_size=(256, 256), use_advanced=True):
    """
    Get training augmentation pipeline using Albumentations.

    Args:
        image_size: Target image size (H, W)
        use_advanced: Whether to use advanced augmentations (elastic, grid distortion)

    Returns:
        Albumentations compose object
    """
    base_transforms = [
        A.Resize(image_size[0], image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0
        ),
        # Color augmentations
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=1.0
            ),
        ], p=0.5),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ]

    if use_advanced:
        advanced_transforms = [
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                p=0.3,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.1,
                p=0.3,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0
            ),
        ]
        # Insert advanced transforms before normalization
        insert_idx = len(base_transforms) - 2  # Before Normalize and ToTensorV2
        for t in advanced_transforms:
            base_transforms.insert(insert_idx, t)

    return A.Compose(
        base_transforms,
        additional_targets={'mask': 'mask'}
    )


def get_validation_augmentation(image_size=(256, 256)):
    """
    Get validation/test augmentation pipeline (no random augmentations).

    Args:
        image_size: Target image size (H, W)

    Returns:
        Albumentations compose object
    """
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})


def get_inference_transform(image_size=(256, 256)):
    """
    Get inference-only transform (just resize and normalize).

    Args:
        image_size: Target image size (H, W)

    Returns:
        Albumentations compose object
    """
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
