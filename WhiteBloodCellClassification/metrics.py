import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


class SegmentationMetrics:
    """Compute segmentation metrics: IoU, Dice, Accuracy, Precision, Recall per class."""

    def __init__(self, num_classes=3, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.reset()

    def reset(self):
        """Reset all accumulated metrics."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_samples = 0

    def update(self, predictions, targets):
        """
        Update metrics with a batch of predictions and targets.

        Args:
            predictions: (B, C, H, W) logits or (B, H, W) class indices
            targets: (B, H, W) ground truth class indices
        """
        if predictions.dim() == 4:
            predictions = torch.argmax(predictions, dim=1)

        predictions = predictions.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()

        # Build confusion matrix
        mask = (targets >= 0) & (targets < self.num_classes)
        hist = np.bincount(
            self.num_classes * targets[mask].astype(int) + predictions[mask].astype(int),
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

        self.confusion_matrix += hist
        self.total_samples += len(targets)

    def compute_iou(self):
        """Compute Intersection over Union for each class."""
        intersection = np.diag(self.confusion_matrix)
        union = self.confusion_matrix.sum(axis=1) + self.confusion_matrix.sum(axis=0) - intersection
        iou = intersection / (union + 1e-10)
        return iou

    def compute_dice(self):
        """Compute Dice coefficient for each class."""
        intersection = np.diag(self.confusion_matrix)
        sum_pred = self.confusion_matrix.sum(axis=1)
        sum_target = self.confusion_matrix.sum(axis=0)
        dice = 2 * intersection / (sum_pred + sum_target + 1e-10)
        return dice

    def compute_accuracy(self):
        """Compute per-class accuracy."""
        acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + 1e-10)
        return acc

    def compute_precision(self):
        """Compute per-class precision."""
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        precision = tp / (tp + fp + 1e-10)
        return precision

    def compute_recall(self):
        """Compute per-class recall."""
        tp = np.diag(self.confusion_matrix)
        fn = self.confusion_matrix.sum(axis=1) - tp
        recall = tp / (tp + fn + 1e-10)
        return recall

    def compute_miou(self):
        """Compute mean Intersection over Union."""
        return np.mean(self.compute_iou())

    def compute_mdice(self):
        """Compute mean Dice coefficient."""
        return np.mean(self.compute_dice())

    def get_results(self):
        """Get all metrics as a dictionary."""
        iou = self.compute_iou()
        dice = self.compute_dice()
        acc = self.compute_accuracy()
        precision = self.compute_precision()
        recall = self.compute_recall()

        results = {
            "mean_iou": float(self.compute_miou()),
            "mean_dice": float(self.compute_mdice()),
            "overall_accuracy": float(np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + 1e-10)),
            "per_class": {}
        }

        for i, name in enumerate(self.class_names):
            results["per_class"][name] = {
                "iou": float(iou[i]),
                "dice": float(dice[i]),
                "accuracy": float(acc[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i])
            }

        return results

    def print_results(self):
        """Print formatted metrics."""
        results = self.get_results()
        print("\n" + "="*60)
        print("Segmentation Metrics")
        print("="*60)
        print(f"Mean IoU: {results['mean_iou']:.4f}")
        print(f"Mean Dice: {results['mean_dice']:.4f}")
        print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
        print("-"*60)
        print(f"{'Class':<15} {'IoU':>8} {'Dice':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8}")
        print("-"*60)
        for name, metrics in results["per_class"].items():
            print(f"{name:<15} {metrics['iou']:>8.4f} {metrics['dice']:>8.4f} "
                  f"{metrics['accuracy']:>8.4f} {metrics['precision']:>8.4f} {metrics['recall']:>8.4f}")
        print("="*60)


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""

    def __init__(self, patience=10, min_delta=0.0, mode='max', verbose=True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics where higher is better (e.g., IoU), 'min' for loss
            verbose: Print messages when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_value = None
        self.early_stop = False

    def __call__(self, value):
        """
        Check if training should stop.

        Args:
            value: Current metric value (e.g., validation mIoU)

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == 'max':
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered!")
                return True
            return False

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_value = None
        self.early_stop = False
