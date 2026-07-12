import os
import sys
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import defaultdict
import random

# Thêm project root vào path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'WhiteBloodCellClassification'))

from DomainAdaptation.TA_module import TaskModule

# ==================== CONFIG ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = r"D:\work\WBC_Segmentation\WhileBloodCellClassification\data\ClassificationData\PBC_dataset_split\Test"
LOG_DIR = r"D:\work\WBC_Segmentation\WhileBloodCellClassification\logs"
RESULTS_DIR = r"D:\work\WBC_Segmentation\WhileBloodCellClassification\results\new"
BATCH_SIZE = 16
NUM_WORKERS = 0
NUM_VISUALIZE = 5
TEST_PERCENTAGE = 1  # CHỈ LẤY 10% TẬP TEST
SEED = 42  # Để kết quả reproducible

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==================== TRANSFORM ====================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ==================== FUNCTION TO GET STRATIFIED SUBSET ====================
def get_stratified_subset(dataset, percentage=0.10, seed=42):
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
    print("\n📊 Class distribution in test subset:")
    for class_id, indices in class_to_indices.items():
        # Số lượng cần lấy từ class này
        n_samples = max(1, int(len(indices) * percentage))
        
        # Random chọn samples
        selected = np.random.choice(indices, n_samples, replace=False)
        selected_indices.extend(selected)
        
        # In thông tin
        print(f"  Class {dataset.classes[class_id]}: {n_samples}/{len(indices)} samples ({n_samples/len(indices)*100:.1f}%)")
    
    # Tạo subset
    subset = Subset(dataset, selected_indices)
    
    print(f"\n✅ Total test subset: {len(subset)}/{len(dataset)} samples ({percentage*100}%)")
    
    return subset

# ==================== LOAD TEST DATA ====================
print("Loading full test dataset...")
full_test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)
class_names = full_test_dataset.classes
print(f"Full test samples: {len(full_test_dataset)} | Classes: {class_names}")

# Lấy 10% stratified subset
print(f"\n{'='*60}")
print(f"CREATING TEST SUBSET WITH {TEST_PERCENTAGE*100}% OF DATA")
print(f"{'='*60}")
test_dataset = get_stratified_subset(full_test_dataset, percentage=TEST_PERCENTAGE, seed=SEED)

# Tạo DataLoader cho subset
test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS
)

# ==================== FIND FOLD_1 CHECKPOINTS ====================
checkpoint_paths = []

# Ưu tiên: checkpoints trong logs/Task_training_CV/*/cv_checkpoints/fold_1/
cv_fold_paths = glob.glob(os.path.join(LOG_DIR, "Task_training_CV", "*", "cv_checkpoints", "fold_1", "*.pth"))
if cv_fold_paths:
    checkpoint_paths.extend(cv_fold_paths)
    print(f"Found {len(cv_fold_paths)} checkpoints in Task_training_CV/*/cv_checkpoints/fold_1/")

# Fallback: logs/fold_1/*.pth
fallback_paths = glob.glob(os.path.join(LOG_DIR, "fold_1", "*.pth"))
if fallback_paths:
    checkpoint_paths.extend(fallback_paths)
    print(f"Found {len(fallback_paths)} checkpoints in logs/fold_1/")

if not checkpoint_paths:
    print("ERROR: No fold_1 checkpoints found! Please run cross-validation training first.")
    print(f"Searched in: {LOG_DIR}")
    sys.exit(1)

# Sort by modification time (newest first)
checkpoint_paths = sorted(checkpoint_paths, key=os.path.getmtime, reverse=True)
print(f"\nTotal checkpoints to evaluate: {len(checkpoint_paths)}")

# ==================== EVALUATION FUNCTION ====================
def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_images = []
    all_paths = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_images.extend(images.cpu())
            
            # Lưu path (cần lấy từ dataset gốc)
            # Vì subset, cần lấy index đúng
            for i in range(len(labels)):
                # Lấy index thực tế trong subset
                idx_in_subset = batch_idx * BATCH_SIZE + i
                if idx_in_subset < len(test_dataset):
                    # Lấy index gốc trong dataset đầy đủ
                    original_idx = test_dataset.indices[idx_in_subset]
                    all_paths.append(full_test_dataset.samples[original_idx][0])
                else:
                    all_paths.append(None)

    return np.array(all_preds), np.array(all_labels), all_images, all_paths

# ==================== VISUALIZATION FUNCTION ====================
def visualize_results(images, labels, preds, class_names, save_path, num=5):
    num = min(num, len(images))
    fig, axes = plt.subplots(1, num, figsize=(num * 3, 3))
    if num == 1:
        axes = [axes]
    for i in range(num):
        img = images[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        color = "green" if labels[i] == preds[i] else "red"
        axes[i].set_title(f"GT: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}", color=color, fontsize=9)
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {save_path}")

# ==================== BENCHMARK ====================
print("\n" + "="*60)
print(f"FOLD 1 BENCHMARK ON {TEST_PERCENTAGE*100}% TEST DATA")
print("="*60)

summary_lines = []
summary_lines.append("="*60)
summary_lines.append(f"FOLD 1 BENCHMARK REPORT (ON {TEST_PERCENTAGE*100}% TEST DATA)")
summary_lines.append("="*60)
summary_lines.append(f"Total test samples used: {len(test_dataset)}/{len(full_test_dataset)} ({TEST_PERCENTAGE*100}%)")
summary_lines.append("")

for ckpt_path in checkpoint_paths:
    ckpt_name = os.path.basename(ckpt_path)
    print(f"\n{'='*60}")
    print(f"Model: {ckpt_name}")
    print(f"Path: {ckpt_path}")
    print(f"{'='*60}")

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model = TaskModule(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Evaluate
    preds, labels, images, paths = evaluate_model(model, test_loader)

    # Metrics
    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, target_names=class_names, digits=4, zero_division=0)
    cm = confusion_matrix(labels, preds)

    print(f"\nOverall Accuracy: {acc:.4f} (on {len(labels)} test samples)")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)

    # Save report
    report_name = ckpt_name.replace('.pth', f'_report_{TEST_PERCENTAGE*100:.0f}percent.txt')
    report_path = os.path.join(RESULTS_DIR, report_name)
    with open(report_path, 'w') as f:
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Test subset size: {len(test_dataset)}/{len(full_test_dataset)} ({TEST_PERCENTAGE*100}%)\n")
        f.write(f"Overall Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(np.array2string(cm, separator=', '))
    print(f"Saved report: {report_path}")

    # Summary
    summary_lines.append(f"\nModel: {ckpt_name}")
    summary_lines.append(f"  Accuracy: {acc:.4f} (on {len(labels)} samples)")
    summary_lines.append(f"  Path: {ckpt_path}")

    # Visualize 5 correct + 5 incorrect examples
    correct_mask = (preds == labels)
    incorrect_mask = ~correct_mask

    # Chọn 5 correct đầu tiên
    correct_indices = np.where(correct_mask)[0]
    if len(correct_indices) > 0:
        selected_correct = correct_indices[:NUM_VISUALIZE]
        vis_images = [images[i] for i in selected_correct]
        vis_labels = [labels[i] for i in selected_correct]
        vis_preds = [preds[i] for i in selected_correct]
        vis_path = os.path.join(RESULTS_DIR, ckpt_name.replace('.pth', f'_correct_{TEST_PERCENTAGE*100:.0f}percent.png'))
        visualize_results(vis_images, vis_labels, vis_preds, class_names, vis_path, num=len(vis_images))

    # Chọn 5 incorrect đầu tiên
    incorrect_indices = np.where(incorrect_mask)[0]
    if len(incorrect_indices) > 0:
        selected_incorrect = incorrect_indices[:NUM_VISUALIZE]
        vis_images = [images[i] for i in selected_incorrect]
        vis_labels = [labels[i] for i in selected_incorrect]
        vis_preds = [preds[i] for i in selected_incorrect]
        vis_path = os.path.join(RESULTS_DIR, ckpt_name.replace('.pth', f'_incorrect_{TEST_PERCENTAGE*100:.0f}percent.png'))
        visualize_results(vis_images, vis_labels, vis_preds, class_names, vis_path, num=len(vis_images))

# Save summary
summary_lines.append("\n" + "="*60)
summary_path = os.path.join(RESULTS_DIR, f"summary_{TEST_PERCENTAGE*100:.0f}percent.txt")
with open(summary_path, 'w') as f:
    f.write('\n'.join(summary_lines))

print(f"\n{'='*60}")
print(f"Benchmark complete!")
print(f"Test subset size: {len(test_dataset)}/{len(full_test_dataset)} ({TEST_PERCENTAGE*100}%)")
print(f"Summary saved to: {summary_path}")
print(f"Results directory: {RESULTS_DIR}")
print(f"{'='*60}")