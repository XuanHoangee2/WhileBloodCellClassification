import os
import yaml
import csv
import json
from datetime import datetime
import torch

class ExperimentLogger:
    def __init__(self, save_dir = "logs"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(save_dir, f"exp_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)

        self.metrics_file = os.path.join(self.exp_dir, "metrics.csv")
        with open(self.metrics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "train_ce_loss",
                "train_mask_loss",
                "train_rec_loss",
                "val_loss",
                "val_miou",
                "val_mdice"
            ])

        self.best_model_path = None
        self.best_metric = float('-inf')

    def log_epoch(self, epoch, train_loss, train_ce, train_mask, train_rec,
                  val_loss=None, val_miou=None, val_mdice=None):
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_loss,
                train_ce,
                train_mask,
                train_rec,
                val_loss if val_loss is not None else "",
                val_miou if val_miou is not None else "",
                val_mdice if val_mdice is not None else ""
            ])

    def save_checkpoint(self, model, epoch, is_best=False):
        """Save model checkpoint."""
        path = os.path.join(self.exp_dir, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), path)

        if is_best:
            best_path = os.path.join(self.exp_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            self.best_model_path = best_path

    def save_metrics_json(self, metrics_dict, filename="final_metrics.json"):
        """Save metrics as JSON."""
        path = os.path.join(self.exp_dir, filename)
        with open(path, "w") as f:
            json.dump(metrics_dict, f, indent=2)

    def log_config(self, config):
        with open(os.path.join(self.exp_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f, sort_keys=False)