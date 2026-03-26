import os
import yaml
import csv
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
                "loss",
                "ce_loss",
                "mask_loss",
                "rec_loss"
            ])

    def log_epoch(self, epoch, loss, ce, mask, rec):
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, loss, ce, mask, rec])
    
    def save_checkpoint(self, model, epoch):
        path = os.path.join(self.exp_dir, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), path)
    

    def log_config(self, config):
        with open(os.path.join(self.exp_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f, sort_keys=False)