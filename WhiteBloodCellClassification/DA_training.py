from tqdm import tqdm
from dataset import JSTCDataset
from torch.utils.data import Dataset, DataLoader
from config.config_loader import load_config
import torch
from DomainAdaptation.DA_module import DomainAdaptationModule
from models.losses import *
from utils import *
from logger import ExperimentLogger
from metrics import SegmentationMetrics, EarlyStopping
import numpy as np
import torch.nn as nn


def training_fn(loader, model, optimizer, CrossEntropyLoss, DICELoss, BCELoss,
                ReconstructionLoss, scaler, device):
    loop = tqdm(loader, desc="Training")
    total_loss = 0
    total_ce = 0
    total_mask = 0
    total_rec = 0
    num_batches = len(loader)

    model.train()

    for batch in loop:
        images = batch["image"]
        masks = batch["mask"]
        images = images.to(device=device)
        masks = masks.float().to(device=device)
        masks_long = masks.long().to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            predictions, encoded_features, query_features = model(images)
            CrossEntropy = CrossEntropyLoss(predictions, masks_long)
            mask_loss = 0
            K = predictions.shape[1]
            for k in range(K):
                pred_binary  = predictions[:, k:k+1, :, :]
                target_binary = (masks == k).float()
                dice = DICELoss(pred_binary, target_binary.long().unsqueeze(1))
                bce = BCELoss(pred_binary, target_binary.unsqueeze(1))
                mask_loss += bce + dice
            mask_loss = mask_loss / K

            Reconstruction = ReconstructionLoss(encoded_features, query_features)
            loss = weight_cross_entropy * CrossEntropy + weight_mask * mask_loss + weight_rec * Reconstruction

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_ce += CrossEntropy.item()
        total_mask += mask_loss.item()
        total_rec += Reconstruction.item()

        # update tqdm loop
        loop.set_postfix(
            total=f"{loss.item():.4f}",
            ce=f"{CrossEntropy.item():.4f}",
            mask=f"{mask_loss.item():.4f}",
            rec=f"{Reconstruction.item():.4f}"
        )

    return (
        total_loss / num_batches,
        total_ce / num_batches,
        total_mask / num_batches,
        total_rec / num_batches
    )


def validation_fn(loader, model, CrossEntropyLoss, DICELoss, BCELoss,
                  ReconstructionLoss, device, metrics_tracker):
    """Validation loop with metrics computation."""
    loop = tqdm(loader, desc="Validation")
    total_loss = 0
    total_ce = 0
    total_mask = 0
    total_rec = 0
    num_batches = len(loader)

    model.eval()
    metrics_tracker.reset()

    with torch.no_grad():
        for batch in loop:
            images = batch["image"]
            masks = batch["mask"]
            images = images.to(device=device)
            masks = masks.float().to(device=device)
            masks_long = masks.long().to(device=device)

            # forward
            with torch.cuda.amp.autocast():
                predictions, encoded_features, query_features = model(images)
                CrossEntropy = CrossEntropyLoss(predictions, masks_long)
                mask_loss = 0
                K = predictions.shape[1]
                for k in range(K):
                    pred_binary  = predictions[:, k:k+1, :, :]
                    target_binary = (masks == k).float()
                    dice = DICELoss(pred_binary, target_binary.long().unsqueeze(1))
                    bce = BCELoss(pred_binary, target_binary.unsqueeze(1))
                    mask_loss += bce + dice
                mask_loss = mask_loss / K

                Reconstruction = ReconstructionLoss(encoded_features, query_features)
                loss = weight_cross_entropy * CrossEntropy + weight_mask * mask_loss + weight_rec * Reconstruction

            total_loss += loss.item()
            total_ce += CrossEntropy.item()
            total_mask += mask_loss.item()
            total_rec += Reconstruction.item()

            # Update metrics
            metrics_tracker.update(predictions, masks_long)

            loop.set_postfix(loss=f"{loss.item():.4f}")

    val_loss = total_loss / num_batches
    results = metrics_tracker.get_results()

    return val_loss, results


config = load_config()
SEED = config["Domain_Adaptation_training"]["SEED"]
logger = ExperimentLogger()
logger.log_config(config)
EPOCHS = config["Domain_Adaptation_training"]["NUM_EPOCHS"]
root_dir = config["dataset"]["root_dir"]
mask_dir = config["dataset"]["mask_dir"]
DEVICE = torch.device(config["Domain_Adaptation_training"]["DEVICE"] if torch.cuda.is_available() else "cpu")
weight_cross_entropy = config["weight"]["weight_cross_entropy"]
weight_mask = config["weight"]["weight_mask"]
weight_rec = config["weight"]["weight_rec"]
weight_boundary = config["weight"]["weight_boundary"]
batch_size = config["Domain_Adaptation_training"]["BATCH_SIZE"]
num_workers = config["Domain_Adaptation_training"].get("NUM_WORKERS", 0)
pin_memory = config["Domain_Adaptation_training"].get("PIN_MEMORY", False)
background_weight = config["class_weights"]["background"]
cytoplasm_weight = config["class_weights"]["cytoplasm"]
nucleus_weight = config["class_weights"]["nucleus"]

np.random.seed(SEED)

# Create datasets with augmentation for training
fulldataset = JSTCDataset(root_dir, mask_dir)
num_samples = len(fulldataset)
indices = np.arange(num_samples)
np.random.shuffle(indices)
split = int(0.8 * num_samples)

train_indices = indices[:split]
val_indices = indices[split:]

# Training dataset with augmentation
train_dataset = JSTCDataset(
    root_dir, mask_dir,
    indices=train_indices,
    use_augmentation=True,
    is_training=True
)

# Validation dataset without augmentation
val_dataset = JSTCDataset(
    root_dir, mask_dir,
    indices=val_indices,
    use_augmentation=False,
    is_training=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory
)

model = DomainAdaptationModule().to(DEVICE)

CrossEntropyLoss = CLSLoss(class_weights=[background_weight, cytoplasm_weight, nucleus_weight])
diceLoss = DiceLoss()
bceLoss = nn.BCEWithLogitsLoss()
reconstructionLoss = ReconstructionLoss()

# Setup optimizer with learning rate scheduler
optimizer = get_optimizer(model, reconstructionLoss)

# Learning rate scheduler - Cosine Annealing with Warmup
scheduler_config = config.get("scheduler", {})
scheduler_type = scheduler_config.get("type", "cosine")
T_max = scheduler_config.get("T_max", EPOCHS)
eta_min = scheduler_config.get("eta_min", 1e-6)
warmup_epochs = scheduler_config.get("warmup_epochs", 5)

if scheduler_type == "cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
elif scheduler_type == "plateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, verbose=True
    )
else:
    scheduler = None

scaler = get_scaler()

# Metrics and early stopping
metrics_tracker = SegmentationMetrics(
    num_classes=3,
    class_names=["background", "cytoplasm", "nucleus"]
)

early_stopping_config = config.get("early_stopping", {})
early_stopping = EarlyStopping(
    patience=early_stopping_config.get("patience", 10),
    min_delta=early_stopping_config.get("min_delta", 0.001),
    mode='max',
    verbose=True
)

best_miou = 0.0

for epoch in range(EPOCHS+1):
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

    # Training
    train_loss, train_ce, train_mask, train_rec = training_fn(
        loader=train_loader,
        model=model,
        optimizer=optimizer,
        CrossEntropyLoss=CrossEntropyLoss,
        DICELoss=diceLoss,
        BCELoss=bceLoss,
        ReconstructionLoss=reconstructionLoss,
        scaler=scaler,
        device=DEVICE
    )

    # Validation
    val_loss, val_results = validation_fn(
        loader=val_loader,
        model=model,
        CrossEntropyLoss=CrossEntropyLoss,
        DICELoss=diceLoss,
        BCELoss=bceLoss,
        ReconstructionLoss=reconstructionLoss,
        device=DEVICE,
        metrics_tracker=metrics_tracker
    )

    val_miou = val_results["mean_iou"]
    val_mdice = val_results["mean_dice"]

    print(f"Train - Loss={train_loss:.4f}, CE={train_ce:.4f}, Mask={train_mask:.4f}, Rec={train_rec:.4f}")
    print(f"Val - Loss={val_loss:.4f}, mIoU={val_miou:.4f}, mDice={val_mdice:.4f}")
    metrics_tracker.print_results()

    # Log epoch
    logger.log_epoch(
        epoch,
        train_loss, train_ce, train_mask, train_rec,
        val_loss, val_miou, val_mdice
    )

    # Save checkpoint
    is_best = val_miou > best_miou
    if is_best:
        best_miou = val_miou
        print(f"New best model! mIoU: {best_miou:.4f}")

    if (epoch) % 5 == 0 or is_best:
        logger.save_checkpoint(model, epoch, is_best=is_best)

    # Save metrics
    if (epoch) % 5 == 0:
        logger.save_metrics_json(val_results, f"metrics_epoch_{epoch}.json")

    # Update learning rate
    if scheduler is not None:
        if scheduler_type == "plateau":
            scheduler.step(val_miou)
        else:
            scheduler.step()

    # Early stopping check
    if early_stopping(val_miou):
        print(f"Early stopping at epoch {epoch+1}")
        logger.save_metrics_json(val_results, "final_metrics.json")
        break

# Save final metrics if training completed without early stopping
if not early_stopping.early_stop:
    val_loss, val_results = validation_fn(
        loader=val_loader,
        model=model,
        CrossEntropyLoss=CrossEntropyLoss,
        DICELoss=diceLoss,
        BCELoss=bceLoss,
        ReconstructionLoss=reconstructionLoss,
        device=DEVICE,
        metrics_tracker=metrics_tracker
    )
    logger.save_metrics_json(val_results, "final_metrics.json")

print(f"\nTraining completed! Best mIoU: {best_miou:.4f}")
print(f"Best model saved at: {logger.best_model_path}")
