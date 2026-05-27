from tqdm import tqdm
from dataset import JSTCDataset
from torch.utils.data import Dataset, DataLoader
from config.config_loader import load_config
import torch
from DomainAdaptation.DA_module import DomainAdaptationModule
from models.losses import *
from utils import *
from logger import ExperimentLogger
import numpy as np
import torch.nn as nn
import os


def training_fn(loader, model, optimizer, CrossEntropyLoss,DICELoss,BCELoss, ReconstructionLoss, scaler):
    loop = tqdm(loader)
    total_loss = 0
    total_ce = 0
    total_mask = 0
    total_rec = 0
    num_batches = len(loader)
    
    for batch in loop:
        images = batch["image"]
        masks = batch["mask"]
        images = images.to(device=DEVICE)
        masks = masks.float().to(device=DEVICE)
        masks_long = masks.long().to(device=DEVICE)

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
                if k == 1: 
                    mask_loss += bce + 3* dice
                else: 
                    mask_loss += bce + dice
            mask_loss = mask_loss / K
            #####
            # boundary_loss = BoundaryLoss(predictions, masks_long)


            ######
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

config = load_config()
SEED = config["Domain_Adaptation_training"]["SEED"]
logger = ExperimentLogger()
logger.log_config(config)
EPOCHS = config["Domain_Adaptation_training"]["NUM_EPOCHS"]
root_dir = config["dataset"]["root_dir"]
mask_dir = config["dataset"]["mask_dir"]
# binary_dir = config["dataset"]["binary_dir"]
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
RESUME_TRAINING = config["Training_Configuration"]["RESUME_TRAINING"]  
CHECKPOINT_PATH = config["Training_Configuration"]["CHECKPOINT_PATH"]
PRETRAINED_WEIGHTS_PATH = config["Training_Configuration"].get("PRETRAINED_WEIGHTS_PATH", None)

# np.random.seed(SEED)

fulldataset = JSTCDataset(root_dir, mask_dir)
num_samples = len(fulldataset)
# indices = np.arange(num_samples)
# np.random.shuffle(indices)
# split = int(0.8 * num_samples)

# train_indices = indices[:split]
# val_indices = indices[split:]
# train_dataset = JSTCDataset(root_dir, mask_dir, indices=train_indices)
# val_dataset   = JSTCDataset(root_dir, mask_dir, indices=val_indices)
# train_loader = DataLoader(
#     train_dataset,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=num_workers,
#     pin_memory=pin_memory
# )

# val_loader = DataLoader(
#     val_dataset,
#     batch_size=batch_size,
#     shuffle=False,
#     num_workers=num_workers,
#     pin_memory=pin_memory
# )
# path_model_trained = "D:\work\WBC_Segmentation\WhileBloodCellClassification\logs\exp_20260402_004902\model_epoch_15.pth"
# model = DomainAdaptationModule().to(DEVICE)
# model.load_state_dict(torch.load(path_model_trained, map_location="cpu"))
# model.eval()
CrossEntropyLoss = CLSLoss(class_weights=[background_weight, cytoplasm_weight, nucleus_weight])
diceLoss = DiceLoss()
bceLoss = nn.BCEWithLogitsLoss()
reconstructionLoss = ReconstructionLoss()

path_model_trained = CHECKPOINT_PATH

model = DomainAdaptationModule(pretrained= not RESUME_TRAINING and PRETRAINED_WEIGHTS_PATH is None).to(DEVICE)
optimizer = get_optimizer(model, reconstructionLoss)
scaler = get_scaler()
start_epoch = 0

if RESUME_TRAINING and os.path.exists(path_model_trained):
    checkpoint = torch.load(path_model_trained, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resume training from epoch {start_epoch}")
elif PRETRAINED_WEIGHTS_PATH and os.path.exists(PRETRAINED_WEIGHTS_PATH):
    # Load pretrained weights from .pth file (weights only, not full checkpoint)
    state_dict = torch.load(PRETRAINED_WEIGHTS_PATH, map_location="cpu")
    # Handle both cases: if the file contains only state_dict or a dict with 'model_state_dict'
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained weights from {PRETRAINED_WEIGHTS_PATH}")
else:
    print("No checkpoint found, starting training from scratch.")

for epoch in range(start_epoch, EPOCHS+1):
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
    epoch_seed = SEED + epoch 
    indices = np.arange(num_samples)
    np.random.seed(epoch_seed)
    np.random.shuffle(indices)
    split = int(0.8 * num_samples)

    train_indices = indices[:split]
    val_indices = indices[split:]
    train_dataset = JSTCDataset(root_dir, mask_dir, indices=train_indices)
    val_dataset   = JSTCDataset(root_dir, mask_dir, indices=val_indices)
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
    loss, ce, mask, rec = training_fn(loader=train_loader, model=model, optimizer=optimizer, CrossEntropyLoss=CrossEntropyLoss, DICELoss=diceLoss, BCELoss=bceLoss, ReconstructionLoss=reconstructionLoss, scaler=scaler)
    print(f"Epoch {epoch+1}: Loss={loss:.4f}, CE={ce:.4f}, Mask={mask:.4f}, Rec={rec:.4f}")

    logger.log_epoch(epoch, loss, ce, mask, rec)
    if (epoch) % 5 == 0:
        logger.save_checkpoint(model, epoch, optimizer, scaler)