from tqdm import tqdm
from DomainAdaptation.TA_module import TaskModule
from models.losses import *
from utils import *
from logger import ExperimentClassificationLogger
import numpy as np
import torch.nn as nn
import os
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def training_classification_fn(loader,model,optimizer,ClassificationLoss, scaler):
    loop = tqdm(loader)
    total_loss = 0
    num_batches = len(loader)
    
    for batch in loop:
        images, labels = batch
        images = images.to(device=DEVICE)
        labels = labels.long().to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(images)
            loss = ClassificationLoss(predictions, labels)

        # backward

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(
            total=f"{loss.item():.4f}",
        )
    return(total_loss/num_batches)

config = load_config(path = "config_task.yaml")
SEED = config["Task_training"]["SEED"]
DEVICE = torch.device(config["Task_training"]["DEVICE"] if torch.cuda.is_available() else "cpu")
logger = ExperimentClassificationLogger(phase = "Task_training")
logger.log_config(config)
EPOCHS = config["Task_training"]["NUM_EPOCHS"]
pin_memory=config["Task_training"]["PIN_MEMORY"]
batch_size = config["Task_training"]["BATCH_SIZE"]
num_workers = config["Task_training"]["NUM_WORKERS"]

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
train_classification_dataset  = datasets.ImageFolder(root=r"D:\work\WBC_Segmentation\WhileBloodCellClassification\data/ClassificationData/PBC_dataset_split/Train", transform=transform)
train_classification_loader = DataLoader(train_classification_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)

Loss = ClassificationLoss()
model = TaskModule(num_classes=8).to(DEVICE)
optimizer = get_classification_optimizer(model, learning_rate=learning_rate, weight_decay=weight_decays)
scaler = get_scaler()
start_epoch = 0
freeze_layers = ['pixel_decoder', 'transformer_decoder']
for name, param in model.named_parameters():
    if any(layer_name in name for layer_name in freeze_layers):
        param.requires_grad = False
        print(f"Freezing layer: {name}")

for epoch in range(start_epoch, EPOCHS+1):
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
    loss = training_classification_fn(loader=train_classification_loader, model=model, optimizer=optimizer, ClassificationLoss=Loss, scaler=scaler)
    print(f"Epoch {epoch+1}: Loss={loss:.4f}")

    logger.log_epoch(epoch, loss)
    if (epoch) % 5 == 0:
        logger.save_checkpoint(model, epoch, optimizer, scaler)
