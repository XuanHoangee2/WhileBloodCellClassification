"""
Inference script for WBC segmentation model.
Supports single image inference, batch inference, and metrics evaluation.
"""

import os
import argparse
import json
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'WhiteBloodCellClassification'))

from WhiteBloodCellClassification.DomainAdaptation.DA_module import DomainAdaptationModule
from WhiteBloodCellClassification.dataset import JSTCDataset
from WhiteBloodCellClassification.metrics import SegmentationMetrics
from WhiteBloodCellClassification.augmentation import get_inference_transform


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    model = DomainAdaptationModule()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def infer_single_image(model, image_path, device='cuda', image_size=(256, 256)):
    """Run inference on a single image."""
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply inference transform
    transform = get_inference_transform(image_size)
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            predictions, _, _ = model(image_tensor)

    # Get predicted classes
    pred_mask = torch.argmax(predictions, dim=1).squeeze(0).cpu().numpy()

    return pred_mask


def visualize_prediction(image_path, pred_mask, output_path=None, alpha=0.5):
    """Create visualization of prediction overlay on original image."""
    # Load original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (pred_mask.shape[1], pred_mask.shape[0]))

    # Create color mask
    color_mask = np.zeros_like(image)
    color_mask[pred_mask == 1] = [0, 255, 0]  # Cytoplasm - Green
    color_mask[pred_mask == 2] = [255, 0, 0]  # Nucleus - Red

    # Blend
    overlay = cv2.addWeighted(image, 1-alpha, color_mask, alpha, 0)

    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return overlay


def save_prediction_mask(pred_mask, output_path):
    """Save prediction mask as PNG with proper label encoding."""
    # Encode labels: 0->0, 1->128, 2->255
    encoded_mask = np.zeros_like(pred_mask, dtype=np.uint8)
    encoded_mask[pred_mask == 1] = 128
    encoded_mask[pred_mask == 2] = 255
    cv2.imwrite(output_path, encoded_mask)


def batch_inference(model, image_dir, output_dir, device='cuda', image_size=(256, 256)):
    """Run inference on all images in a directory."""
    os.makedirs(output_dir, exist_ok=True)

    image_files = list(Path(image_dir).glob('*.bmp')) + list(Path(image_dir).glob('*.png')) + list(Path(image_dir).glob('*.jpg'))

    print(f"Found {len(image_files)} images for inference")

    for img_path in tqdm(image_files, desc="Inference"):
        pred_mask = infer_single_image(model, str(img_path), device, image_size)

        # Save prediction mask
        mask_output = os.path.join(output_dir, f"{img_path.stem}_mask.png")
        save_prediction_mask(pred_mask, mask_output)

        # Save visualization
        vis_output = os.path.join(output_dir, f"{img_path.stem}_overlay.png")
        visualize_prediction(str(img_path), pred_mask, vis_output)

    print(f"Results saved to {output_dir}")


def evaluate_model(model, dataset, device='cuda', batch_size=8):
    """Evaluate model on a dataset and return metrics."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    metrics = SegmentationMetrics(num_classes=3, class_names=["background", "cytoplasm", "nucleus"])

    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluation"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            with torch.cuda.amp.autocast():
                predictions, _, _ = model(images)

            metrics.update(predictions, masks)

    results = metrics.get_results()
    metrics.print_results()

    return results


def main():
    parser = argparse.ArgumentParser(description="WBC Segmentation Inference")
    parser.add_argument("--mode", choices=["single", "batch", "eval"], required=True,
                        help="Inference mode: single image, batch, or evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True,
                        help="Input image path (single) or directory (batch/eval)")
    parser.add_argument("--output", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--image-size", type=int, nargs=2, default=[256, 256],
                        help="Image size (H W)")
    parser.add_argument("--mask-dir", type=str, default=None,
                        help="Mask directory for evaluation mode")

    args = parser.parse_args()

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    if args.mode == "single":
        # Single image inference
        pred_mask = infer_single_image(model, args.input, device, tuple(args.image_size))

        os.makedirs(args.output, exist_ok=True)
        output_mask = os.path.join(args.output, "prediction_mask.png")
        output_vis = os.path.join(args.output, "prediction_overlay.png")

        save_prediction_mask(pred_mask, output_mask)
        visualize_prediction(args.input, pred_mask, output_vis)

        print(f"Results saved to {args.output}")

    elif args.mode == "batch":
        # Batch inference
        batch_inference(model, args.input, args.output, device, tuple(args.image_size))

    elif args.mode == "eval":
        # Evaluation mode
        if args.mask_dir is None:
            print("Error: --mask-dir required for evaluation mode")
            return

        from WhiteBloodCellClassification.dataset import JSTCDataset
        dataset = JSTCDataset(args.input, args.mask_dir, use_augmentation=False, is_training=False)
        results = evaluate_model(model, dataset, device)

        # Save metrics
        metrics_file = os.path.join(args.output, "evaluation_metrics.json")
        os.makedirs(args.output, exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Metrics saved to {metrics_file}")


if __name__ == "__main__":
    main()
