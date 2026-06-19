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
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from sklearn.model_selection import KFold
import random
from collections import defaultdict
import shutil

def get_stratified_subset(dataset, percentage=0.05, seed=42):
    """
    Lấy subset với tỷ lệ percentage nhưng vẫn giữ nguyên phân phối classes
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Lấy labels của tất cả samples
    labels = [label for _, label in dataset.samples]
    
    # Gom indices theo từng class
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_to_indices[label].append(idx)
    
    # Chọn samples từ mỗi class theo tỷ lệ percentage
    selected_indices = []
    print("\n📊 Class distribution in subset:")
    for class_id, indices in class_to_indices.items():
        # Số lượng cần lấy từ class này
        n_samples = max(1, int(len(indices) * percentage))
        
        # Random chọn samples
        selected = np.random.choice(indices, n_samples, replace=False)
        selected_indices.extend(selected)
        
        # In thông tin
        print(f"  Class {class_id}: {n_samples}/{len(indices)} samples ({n_samples/len(indices)*100:.1f}%)")
    
    # Tạo subset
    subset = Subset(dataset, selected_indices)
    
    print(f"\n✅ Total: {len(subset)}/{len(dataset)} samples ({percentage*100}%)")
    
    return subset

def load_pretrained_model(model, pretrained_path, device='cpu'):
    """
    Load pretrained weights from checkpoint file
    """
    if not os.path.exists(pretrained_path):
        print(f"⚠️  Pretrained weights file not found: {pretrained_path}")
        return model, 0
    
    print(f"📥 Loading pretrained weights from: {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', 0)
            print(f"   Loaded checkpoint from epoch {epoch}")
        else:
            state_dict = checkpoint
            epoch = 0
    else:
        state_dict = checkpoint
        epoch = 0
    
    # Load state dict with strict=False to allow missing keys
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"   Missing keys: {len(missing_keys)} keys")
    if unexpected_keys:
        print(f"   Unexpected keys: {len(unexpected_keys)} keys")
    
    print(f"✅ Successfully loaded pretrained weights")
    return model, epoch

def training_classification_fn(loader, model, optimizer, ClassificationLoss, scaler, use_cuda=True):
    loop = tqdm(loader)
    total_loss = 0
    num_batches = len(loader)
    
    for batch in loop:
        images, labels = batch
        images = images.to(device=DEVICE)
        labels = labels.long().to(device=DEVICE)

        # Only use autocast if CUDA is available
        if use_cuda:
            with torch.cuda.amp.autocast():
                predictions = model(images)
                loss = ClassificationLoss(predictions, labels)
        else:
            predictions = model(images)
            loss = ClassificationLoss(predictions, labels)

        optimizer.zero_grad()
        
        if use_cuda and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(total=f"{loss.item():.4f}")
    
    return total_loss / num_batches

def validation_classification_fn(loader, model, ClassificationLoss, use_cuda=True):
    model.eval()
    total_loss = 0
    num_batches = len(loader)
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            images, labels = batch
            images = images.to(device=DEVICE)
            labels = labels.long().to(device=DEVICE)
            
            if use_cuda:
                with torch.cuda.amp.autocast():
                    predictions = model(images)
                    loss = ClassificationLoss(predictions, labels)
            else:
                predictions = model(images)
                loss = ClassificationLoss(predictions, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(predictions, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    return total_loss / num_batches, accuracy

def train_fold(fold, train_loader, val_loader, config, logger, use_cuda=True, 
               resume_training=False, pretrained_path=None, domain_adaptation_path = None):
    print(f"\n{'='*50}")
    print(f"Training Fold {fold + 1}/{config['Task_training']['NUM_FOLDS']}")
    print(f"{'='*50}")
    
    drive_fold_path = f'/PBC_dataset_split/MyDrive/Colab Notebooks/fold_{fold+1}'
    os.makedirs(drive_fold_path, exist_ok=True)
    model = TaskModule(num_classes=8).to(DEVICE)
    
    freeze_layers = ['pixel_decoder', 'transformer_decoder']
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in freeze_layers):
            param.requires_grad = False
    
    learning_rate = config["Task_training"]["LEARNING_RATE"]
    weight_decays = config["Task_training"]["WEIGHT_DECAY"]
    optimizer = get_classification_optimizer(model, learning_rate=learning_rate, weight_decay=weight_decays)
    
    # Only create scaler if CUDA is available
    scaler = None
    if use_cuda:
        scaler = get_scaler()
    
    Loss = ClassificationLoss()
    
    EPOCHS = config["Task_training"]["NUM_EPOCHS"]
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    
    # Load pretrained weights or resume training
    if resume_training and pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint.get('epoch', -1) + 1
        best_val_accuracy = checkpoint.get('best_accuracy', 0.0)
        best_val_loss = checkpoint.get('best_loss', float('inf'))
        print(f"🔄 Resuming training from epoch {start_epoch}")
        print(f"   Best val accuracy so far: {best_val_accuracy:.4f}")
    elif pretrained_path and os.path.exists(pretrained_path) and not resume_training:
        model, _ = load_pretrained_model(model, pretrained_path, DEVICE)
        print(f"📥 Using pretrained weights, starting from scratch")
    else:
        print(f"✨ Starting training from scratch")
        if domain_adaptation_path is None:
            print(f"✨ No domain adaptation weights provided")
        else:
            checkpoint = torch.load(domain_adaptation_path, map_location="cpu")
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded domain adaptation weights from {domain_adaptation_path}")
            if missing:
                print(f"   Missing keys (expected for classifier head): {len(missing)}")
            if unexpected:
                print(f"   Unexpected keys: {len(unexpected)}")
    
    patience = config["Task_training"].get("PATIENCE", 5)
    patience_counter = 0
    
    for epoch in range(start_epoch, EPOCHS):
        print(f"\nFold {fold + 1} - Epoch [{epoch+1}/{EPOCHS}]")
        
        model.train()
        train_loss = training_classification_fn(
            loader=train_loader, 
            model=model, 
            optimizer=optimizer, 
            ClassificationLoss=Loss, 
            scaler=scaler,
            use_cuda=use_cuda
        )
        
        val_loss, val_accuracy = validation_classification_fn(
            loader=val_loader, 
            model=model, 
            ClassificationLoss=Loss,
            use_cuda=use_cuda
        )
        
        print(f"Fold {fold + 1} - Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Accuracy={val_accuracy:.4f}")
        
        # Log metrics using the logger
        logger.log_metrics_cv(fold, epoch, train_loss, val_loss, val_accuracy)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            patience_counter = 0
            logger.save_fold_checkpoint(model, fold, epoch, optimizer, scaler, best_val_accuracy)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        if (epoch + 1) % 2 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }
            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            
            torch.save(checkpoint, f'{drive_fold_path}/checkpoint_epoch_{epoch+1}.pth')
            print(f"💾 Checkpoint for fold {fold+1} epoch {epoch+1} saved to Drive")
    
    return best_val_accuracy, best_val_loss

def cross_validation_training(config, dataset, use_cuda=True, pretrained_path=None, resume_training=False,domain_adaptation_path = None):
    SEED = config["Task_training"]["SEED"]
    NUM_FOLDS = config["Task_training"].get("NUM_FOLDS", 5)
    batch_size = config["Task_training"]["BATCH_SIZE"]
    pin_memory = config["Task_training"]["PIN_MEMORY"]
    num_workers = config["Task_training"]["NUM_WORKERS"]
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    
    fold_results = []
    
    # Main logger for cross-validation
    cv_logger = ExperimentClassificationLogger(phase="Task_training_CV")
    cv_logger.log_config(config)
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(indices)):
        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            sampler=train_sampler,
            pin_memory=pin_memory, 
            num_workers=num_workers
        )
        
        val_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            sampler=val_sampler,
            pin_memory=pin_memory, 
            num_workers=num_workers
        )
        
        print(f"\nFold {fold + 1}:")
        print(f"Training samples: {len(train_ids)}")
        print(f"Validation samples: {len(val_ids)}")
        
        best_accuracy, best_loss = train_fold(
            fold, train_loader, val_loader, config, cv_logger, use_cuda,
            resume_training=resume_training, pretrained_path=pretrained_path, domain_adaptation_path=domain_adaptation_path
        )
        
        fold_results.append({
            'fold': fold + 1,
            'best_accuracy': best_accuracy,
            'best_loss': best_loss
        })
    
    # Calculate and log final results
    accuracies = [result['best_accuracy'] for result in fold_results]
    losses = [result['best_loss'] for result in fold_results]
    
    cv_logger.log_cv_results(accuracies, losses)
    
    # Find and log best fold
    best_fold_idx = np.argmax(accuracies)
    cv_logger.log_best_fold_summary(best_fold_idx, accuracies[best_fold_idx])
    
    return fold_results

def train_final_model(config, dataset, use_cuda=True, pretrained_path=None, resume_training=False,domain_adaptation_path = None):
    """Train final model on full dataset"""
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL")
    print("="*60)

    drive_models_path = '/PBC_dataset_split/MyDrive/Colab Notebooks'
    os.makedirs(drive_models_path, exist_ok=True)
    print(f"📁 Models will be saved to: {drive_models_path}")

    train_loader = DataLoader(
        dataset, 
        batch_size=config["Task_training"]["BATCH_SIZE"], 
        shuffle=True, 
        pin_memory=config["Task_training"]["PIN_MEMORY"], 
        num_workers=config["Task_training"]["NUM_WORKERS"]
    )
    
    model = TaskModule(num_classes=8).to(DEVICE)
    
    freeze_layers = ['pixel_decoder', 'transformer_decoder']
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in freeze_layers):
            param.requires_grad = False
    
    optimizer = get_classification_optimizer(
        model, 
        learning_rate=config["Task_training"]["LEARNING_RATE"], 
        weight_decay=config["Task_training"]["WEIGHT_DECAY"]
    )
    
    # Only create scaler if CUDA is available
    scaler = None
    if use_cuda:
        scaler = get_scaler()
    
    Loss = ClassificationLoss()
    
    final_logger = ExperimentClassificationLogger(phase="Task_training_Final")
    final_logger.log_config(config)
    
    EPOCHS = config["Task_training"]["NUM_EPOCHS"]
    start_epoch = 0
    best_loss = float('inf')
    
    # Load pretrained weights or resume training
    if resume_training and pretrained_path and os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint.get('epoch', -1) + 1
        best_loss = checkpoint.get('loss', float('inf'))
        print(f"🔄 Resuming final model training from epoch {start_epoch}")
    elif pretrained_path and os.path.exists(pretrained_path) and not resume_training:
        model, _ = load_pretrained_model(model, pretrained_path, DEVICE)
        print(f"📥 Using pretrained weights for final model")
    else:
        print("Training from scratch")
    
    for epoch in range(start_epoch, EPOCHS):
        print(f"\nFinal Model - Epoch [{epoch+1}/{EPOCHS}]")
        model.train()
        loss = training_classification_fn(
            loader=train_loader, 
            model=model, 
            optimizer=optimizer, 
            ClassificationLoss=Loss, 
            scaler=scaler,
            use_cuda=use_cuda
        )
        print(f"Final Model - Epoch {epoch+1}: Loss={loss:.4f}")
        
        final_logger.log_epoch(epoch, loss)
        
        if loss < best_loss:
            best_loss = loss
            final_logger.save_checkpoint(model, epoch, optimizer, scaler)
            # Lưu bản copy best model lên Drive
            best_checkpoint_path = f'{drive_models_path}/best_model_epoch{epoch+1}_loss{loss:.4f}.pth'
            shutil.copy2(final_logger.checkpoint_path, best_checkpoint_path)
            print(f"💾 Best model saved to Drive: {best_checkpoint_path}")
        if (epoch + 1) % 2 == 0:
            drive_checkpoint_path = f'{drive_models_path}/checkpoint_epoch_{epoch+1}.pth'
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'best_loss': best_loss,
            }
            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            
            torch.save(checkpoint, drive_checkpoint_path)
            print(f"💾 Checkpoint saved to Drive: {drive_checkpoint_path}")
            final_logger.save_checkpoint(model, epoch, optimizer, scaler)
    
    final_model_path = f'{drive_models_path}/final_model_epoch{EPOCHS}.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"💾 Final model saved to Drive: {final_model_path}")
    return model

# Helper function to check CUDA availability
def is_cuda_available():
    return torch.cuda.is_available()

# Main execution
if __name__ == "__main__":
    config = load_config(path="config_task.yaml")
    
    # Add default CV parameters if not exists
    if "NUM_FOLDS" not in config["Task_training"]:
        config["Task_training"]["NUM_FOLDS"] = 5
    if "PATIENCE" not in config["Task_training"]:
        config["Task_training"]["PATIENCE"] = 5
    if "LEARNING_RATE" not in config["Task_training"]:
        config["Task_training"]["LEARNING_RATE"] = 0.001
    if "WEIGHT_DECAY" not in config["Task_training"]:
        config["Task_training"]["WEIGHT_DECAY"] = 0.0001
    
    # ============ CONFIGURATION ============
    # Subset configuration
    USE_SUBSET = True  # Set to True to use only a percentage of data
    SUBSET_PERCENTAGE = 0.3  # Use 30% of data
    
    # Pretrained model configuration
    PRETRAINED_WEIGHTS_PATH = config["Training_Configuration"]["CHECKPOINT_PATH"] # Path to pretrained weights
    RESUME_TRAINING = config["Training_Configuration"]["RESUME_TRAINING"]
    DOMAIN_ADAPTATION = config["PRETRAIN_PARAMETERS"]["PATH"]  # Set to True to resume training from checkpoint
    # ======================================
    
    # Check CUDA availability
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device(config["Task_training"]["DEVICE"] if USE_CUDA else "cpu")
    
    print(f"\n{'='*60}")
    print(f"CONFIGURATION")
    print(f"{'='*60}")
    print(f"Using device: {DEVICE}")
    print(f"CUDA available: {USE_CUDA}")
    print(f"Using subset: {USE_SUBSET}")
    if USE_SUBSET:
        print(f"Subset percentage: {SUBSET_PERCENTAGE*100}%")
    print(f"Resume training: {RESUME_TRAINING}")
    if PRETRAINED_WEIGHTS_PATH and os.path.exists(PRETRAINED_WEIGHTS_PATH):
        print(f"Pretrained weights: {PRETRAINED_WEIGHTS_PATH}")
    elif PRETRAINED_WEIGHTS_PATH:
        print(f"Pretrained weights path set but file not found: {PRETRAINED_WEIGHTS_PATH}")
    print(f"{'='*60}\n")
    
    if not USE_CUDA:
        print("⚠️  Warning: CUDA is not available. Training will use CPU which may be slow.")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # Load full dataset
    print("Loading dataset...")
    full_dataset = datasets.ImageFolder(
        root=r"/PBC_dataset_split/MyDrive/Colab Notebooks/PBC_dataset_split/Train", 
        transform=transform
    )
    
    print(f"✅ Full dataset loaded: {len(full_dataset)} images, {len(full_dataset.classes)} classes")
    
    # Apply subset if configured
    if USE_SUBSET:
        print(f"\n{'='*60}")
        print(f"CREATING SUBSET WITH {SUBSET_PERCENTAGE*100}% OF DATA")
        print(f"{'='*60}")
        train_dataset = get_stratified_subset(full_dataset, percentage=SUBSET_PERCENTAGE, seed=config["Task_training"]["SEED"])
    else:
        train_dataset = full_dataset
        print(f"\n✅ Using full dataset: {len(train_dataset)} images")
    
    # Adjust settings for CPU if needed
    if not USE_CUDA:
        print("\n⚙️  Adjusting settings for CPU training...")
        if config["Task_training"]["BATCH_SIZE"] > 16:
            config["Task_training"]["BATCH_SIZE"] = 16
            print(f"  - Batch size reduced to: {config['Task_training']['BATCH_SIZE']}")
        if config["Task_training"]["NUM_WORKERS"] > 2:
            config["Task_training"]["NUM_WORKERS"] = 2
            print(f"  - Num workers reduced to: {config['Task_training']['NUM_WORKERS']}")
        config["Task_training"]["PIN_MEMORY"] = False
        print(f"  - Pin memory set to: {config['Task_training']['PIN_MEMORY']}")
    
    # Run cross-validation
    print(f"\n{'='*60}")
    print(f"STARTING CROSS-VALIDATION")
    print(f"{'='*60}")
    
    cv_results = cross_validation_training(
        config, train_dataset, use_cuda=USE_CUDA,
        pretrained_path=PRETRAINED_WEIGHTS_PATH,
        resume_training=RESUME_TRAINING,
        domain_adaptation_path=DOMAIN_ADAPTATION
    )
    
    # Print cross-validation summary
    print("\n" + "="*60)
    print("CROSS-VALIDATION SUMMARY")
    print("="*60)
    for result in cv_results:
        print(f"Fold {result['fold']}: Best Accuracy = {result['best_accuracy']:.4f}, Best Loss = {result['best_loss']:.4f}")
    
    # Calculate average
    avg_accuracy = np.mean([r['best_accuracy'] for r in cv_results])
    avg_loss = np.mean([r['best_loss'] for r in cv_results])
    print(f"\n📊 Average across all folds: Accuracy = {avg_accuracy:.4f}, Loss = {avg_loss:.4f}")
    
    # Train final model
    TRAIN_FINAL_ON_FULL = False  # Set to True to train final model on full dataset
    
    if TRAIN_FINAL_ON_FULL and USE_SUBSET:
        print(f"\n⚠️  Training final model on FULL dataset ({len(full_dataset)} images)")
        final_train_dataset = full_dataset
    else:
        print(f"\n✅ Training final model on current dataset ({len(train_dataset)} images)")
        final_train_dataset = train_dataset
    
    final_model = train_final_model(
        config, final_train_dataset,
        use_cuda=USE_CUDA,
        pretrained_path=PRETRAINED_WEIGHTS_PATH,
        resume_training=RESUME_TRAINING,
        domain_adaptation_path=DOMAIN_ADAPTATION
    )
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)


