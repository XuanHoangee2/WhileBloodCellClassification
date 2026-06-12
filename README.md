# Phân loại Tế bào Máu Trắng (White Blood Cell Classification) với Domain Adaptation

Dự án thực hiện phân loại 8 loại tế bào máu trắng (WBC) bằng kiến trúc **Segmentation-Guided Classification** kết hợp **Domain Adaptation**. Hệ thống được huấn luyện theo 2 giai đoạn:

1. **Domain Adaptation (DA)**: Pretrain mô hình segmentation trên bộ dữ liệu **JSTC** (320 ảnh có nhãn pixel-level: nucleus, cytoplasm, background).
2. **Task Adaptation (TA)**: Fine-tune phân loại trên bộ dữ liệu **PBC** (Peripheral Blood Cell) với 8 lớp tế bào.

---

## Mục lục

- [1. Kiến trúc tổng quan](#1-kiến-trúc-tổng-quan)
- [2. Domain Adaptation Training](#2-domain-adaptation-training)
- [3. Task Adaptation Training](#3-task-adaptation-training)
- [4. Kết quả thử nghiệm](#4-kết-quả-thử-nghiệm)
- [5. Cấu trúc thư mục](#5-cấu-trúc-thư-mục)
- [6. Hướng dẫn chạy](#6-hướng-dẫn-chạy)
- [7. Cấu hình](#7-cấu-hình)

---

## 1. Kiến trúc tổng quan

### Pipeline 2 giai đoạn

```
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: Domain Adaptation (JSTC Dataset)                  │
│  ├─ Input: Ảnh vi thầu (microscopy) + Mask pixel-level     │
│  ├─ Output: Mô hình segmentation (nucleus, cytoplasm, bg)    │
│  └─ Loss: CE + Dice/BCE + Reconstruction                     │
│           ↓ save checkpoint                                  │
│  STAGE 2: Task Adaptation (PBC Dataset)                      │
│  ├─ Input: Ảnh WBC (8 classes)                              │
│  ├─ Load DA weights → Freeze decoder/segmentation            │
│  ├─ Train: Encoder + NuclearCytoplasmicClassifier            │
│  └─ Output: 8-class WBC classifier                         │
└─────────────────────────────────────────────────────────────┘
```

### Các thành phần chính

| Module | Chức năng | Chi tiết |
|--------|-----------|----------|
| **PixelEncoder** | Backbone trích xuất đặc trưng | ResNet-50 pretrained ImageNet, output `[c1, c2, c3, c4]` |
| **PixelDecoder** | FPN-style upsampling | Lateral 1×1 + Bilinear upsample + ConvBlock, output 256-ch |
| **SCFEModule** | Spatial Co-occurrence Query | MLP bottleneck + CoLLayer (cosine similarity) + AvgPool → query `[B,1,256]` |
| **TransformerDecoder** | Refine mask queries | 6 layers, 8 heads, 32 learnable queries. Query 0 được inject SCFE query |
| **SegmentationHead** | Tổng hợp mask | Per-query mask embedding × pixel feature, softmax với temperature=0.5 |
| **NuclearCytoplasmicClassifier** | Phân loại WBC | Weighted spatial pooling của encoder features theo segmentation mask, MLP classifier |

### NuclearCytoplasmicClassifier (Knowledge-aware)

Điểm độc đáo của kiến trúc là **classifier kết hợp thông tin hình thái học** (nucleus + cytoplasm):

- Downsample segmentation mask về độ phân giải của encoder feature (`c4`: 8×8)
- Sigmoid cho phép nucleus và cytoplasm overlap (giống thực tế)
- Weighted pooling: `class_features = torch.einsum('bchw,bkhw->bkc', f_conv, z_prob)`
- MLP: `2048 → 256 → num_classes`

---

## 2. Domain Adaptation Training

### Dataset: JSTC (Japan Society of Technology for Cell)

- **320 ảnh** vi thầu (microscopy) đã gán nhãn pixel-level
- **3 lớp segmentation**: Background (0), Cytoplasm (1), Nucleus (2)
- Mask format: PNG với giá trị 0/128/255
- **Augmentation**: HorizontalFlip, VerticalFlip, Rotation (±15°), ColorJitter (brightness, contrast, saturation)

### Loss Function

```
L_total = 1.0 × L_CE + 2.0 × L_mask + 2.0 × L_reconstruction
```

| Thành phần | Công thức | Trọng số | Mô tả |
|------------|-----------|----------|-------|
| **CrossEntropy** | `F.cross_entropy` với class weights `[1.0, 8.0, 4.0]` | 1.0 | Xử lý class imbalance (cytoplasm nhiều nhất, background ít nhất) |
| **Mask Loss** | Per-class BCE + Dice | 2.0 | Cytoplasm class được nhân 3× Dice weight: `bce + 3*dice` |
| **Reconstruction** | `1 - cosine_similarity(query_0, c3_feature)` | 2.0 | Ép query 0 tái tạo được encoder feature, hoạt động như bottleneck |

### Hyperparameters (config.yaml)

```yaml
Domain_Adaptation_training:
  LEARNING_RATE: 0.0002
  BATCH_SIZE: 8
  NUM_EPOCHS: 30
  WEIGHT_DECAY: 0.001
  SEED: 140

class_weights:
  background: 1.0    # Ít quan trọng
  cytoplasm: 8.0     # Cần focus nhiều nhất
  nucleus: 4.0       # Quan trọng trung bình

weight:
  weight_cross_entropy: 1
  weight_mask: 2
  weight_rec: 2
```

### Chạy DA Training

```bash
cd WhiteBloodCellClassification
python DA_training.py
```

---

## 3. Task Adaptation Training

### Dataset: PBC (Peripheral Blood Cell)

- **Train**: ~1700+ ảnh (8 classes)
- **Test**: 2048 ảnh
- **8 lớp WBC**: `basophil`, `eosinophil`, `erythroblast`, `ig`, `lymphocyte`, `monocyte`, `neutrophil`, `platelet`
- **Transform**: Resize(256,256) + ToTensor
- **Subset**: Có thể chạy với 30% data để test nhanh (`USE_SUBSET = True`)

### Cross-Validation

- **Stratified K-Fold**: `NUM_FOLDS = 5` (giữ tỷ lệ class mỗi fold)
- **Early stopping**: `PATIENCE = 5` epochs
- **Mỗi fold**: Train 80%, Validation 20%

### Freeze Strategy

```python
freeze_layers = ['pixel_decoder', 'transformer_decoder', 'scfe', 'segmentation_head']
```

Các nhánh segmentation được **đóng băng hoàn toàn**, chỉ optimize:
- `pixel_encoder` (backbone ResNet-50) với LR = 0.5 × classifier LR
- `classifier` (NuclearCytoplasmicClassifier) với LR = 0.00005

### Hyperparameters (config_task.yaml)

```yaml
Task_training:
  LEARNING_RATE: 0.00005
  BATCH_SIZE: 8
  NUM_EPOCHS: 30
  NUM_FOLDS: 5
  PATIENCE: 5
  WEIGHT_DECAY: 0.001
  SEED: 2004

Encoder_lr:
  weight: 0.5   # Encoder LR = 0.5 × 0.00005 = 0.000025
```

### Chạy TA Training

```bash
cd WhiteBloodCellClassification
python TA_training.py
```

Pipeline:
1. Cross-Validation (5 folds) để đánh giá độ tin cậy và chọn hyperparameters
2. `train_final_model()` train trên 100% dataset (hoặc subset nếu `USE_SUBSET=True`) với validation split riêng để early stopping

---

## 4. Kết quả thử nghiệm

### 4.1. Domain Adaptation (Segmentation)

| Metric | Giá trị |
|--------|---------|
| Dataset | JSTC (320 ảnh) |
| Classes | 3 (background, cytoplasm, nucleus) |
| Loss weights | CE: 1.0, Mask: 2.0, Rec: 2.0 |
| Training Epochs | 20 epochs |
| Checkpoint | `logs/exp_20260526_211124/model_epoch_20.pth` |

Kết quả segmentation đạt chất lượng tốt với **cytoplasm được up-weight 8.0** và **nucleus 4.0**, giúp mô hình tập trung vào vùng tế bào thay vì background.

### 4.2. Task Adaptation (Classification)

#### Kết quả Test trên PBC Dataset (10% subset = 201 samples)

| Checkpoint | Accuracy |
|------------|----------|
| `checkpoint_epoch_2.pth` | **98.51%** |
| `checkpoint_epoch_5.pth` | **97.51%** |
| `checkpoint_epoch_6.pth` | **97.01%** |
| `checkpoint_epoch_4.pth` | 95.02% |

#### Classification Report (checkpoint_epoch_6.pth — 201 samples)

```
              precision    recall  f1-score   support

    basophil     1.0000    0.9286    0.9630        14
  eosinophil     1.0000    1.0000    1.0000        37
erythroblast     1.0000    0.9444    0.9714        18
          ig     0.9429    0.9706    0.9565        34
  lymphocyte     0.8750    1.0000    0.9333        14
    monocyte     0.8889    0.9412    0.9143        17
  neutrophil     1.0000    0.9487    0.9737        39
    platelet     1.0000    1.0000    1.0000        28

    accuracy                         0.9701       201
   macro avg     0.9633    0.9667    0.9640       201
weighted avg     0.9722    0.9701    0.9705       201
```

#### Confusion Matrix

```
[[13,  0,  0,  1,  0,  0,  0,  0],   # basophil
 [ 0, 37,  0,  0,  0,  0,  0,  0],   # eosinophil
 [ 0,  0, 17,  0,  1,  0,  0,  0],   # erythroblast
 [ 0,  0,  0, 33,  0,  1,  0,  0],   # ig
 [ 0,  0,  0,  0, 14,  0,  0,  0],   # lymphocyte
 [ 0,  0,  0,  0,  1, 16,  0,  0],   # monocyte
 [ 0,  0,  0,  1,  0,  1, 37,  0],   # neutrophil
 [ 0,  0,  0,  0,  0,  0,  0, 28]]   # platelet
```

**Nhận xét:**
- Các lớp **eosinophil** và **platelet** đạt 100% precision/recall — đặc trưng hình thái rất rõ (hạt/láp lánh)
- **Lymphocyte** đạt 100% recall nhưng precision 87.5% → có một số mẫu khác bị nhầm thành lymphocyte
- **Monocyte** và **Neutrophil** có confusion nhẹ với nhau (2 mẫu) — cả hai đều có nucleus đa dạng
- **Macro avg F1 = 0.9640**, **Weighted avg F1 = 0.9705**

---

## 5. Cấu trúc thư mục

```
WhileBloodCellClassification/
├── WhiteBloodCellClassification/
│   ├── models/
│   │   ├── PixelEncoder.py          # ResNet-50 backbone
│   │   ├── PixelDecoder.py        # FPN decoder
│   │   ├── TransformerDecoder.py  # 6-layer transformer decoder
│   │   ├── segmentationHead.py    # Segmentation head
│   │   ├── spatial_cooccurrence.py # SCFE module
│   │   ├── classifier.py           # NuclearCytoplasmicClassifier
│   │   ├── losses.py               # CE, Dice, BCE, Reconstruction, Boundary
│   │   └── blocks.py               # ConvBlock, MLPLayer, CoLLayer
│   ├── DomainAdaptation/
│   │   ├── DA_module.py            # DomainAdaptationModule
│   │   └── TA_module.py            # TaskModule
│   ├── config/
│   │   ├── config.yaml             # DA config (JSTC)
│   │   └── config_task.yaml        # TA config (PBC)
│   ├── dataset.py                  # JSTCDataset
│   ├── DA_training.py             # Domain Adaptation training
│   ├── TA_training.py             # Task Adaptation + Cross-Validation
│   ├── utils.py                    # Optimizers, scaler, checkpoint helpers
│   └── logger.py                   # ExperimentLogger, ExperimentClassificationLogger
├── data/
│   ├── SourceData/                  # JSTC dataset (320 ảnh segmentation)
│   └── ClassificationData/
│       └── PBC_dataset_split/
│           ├── Train/               # ~1700 ảnh train
│           ├── Test/                # 2048 ảnh test
│           └── Val/
├── logs/                            # Checkpoints & logs
│   ├── exp_*/                       # DA training logs
│   ├── Task_training_CV/            # CV logs
│   └── fold_1/                      # Fold checkpoints
├── results/                         # Benchmark outputs
│   └── fold1_benchmark_10percent/
│       ├── *_correct.png            # Visualization đúng
│       ├── *_incorrect.png          # Visualization sai
│       ├── *_report.txt             # Classification report
│       └── summary.txt              # Summary
└── test_fold1.py                    # Script benchmark fold_1
```

---

## 6. Hướng dẫn chạy

### Bước 1: Chuẩn bị môi trường

```bash
pip install torch torchvision torchaudio
pip install opencv-python matplotlib scikit-learn tqdm albumentations
```

### Bước 2: Cấu hình đường dẫn dataset

Sửa file `config.yaml` và `config_task.yaml` cho đúng đường dẫn:

```yaml
# config.yaml
dataset:
  root_dir: "D:/work/.../data/SourceData"
  mask_dir: "Dataset"

# config_task.yaml
dataset:
  root_dir: "D:/work/.../data/ClassificationData"
```

### Bước 3: Chạy Domain Adaptation

```bash
cd WhiteBloodCellClassification
python DA_training.py
```

### Bước 4: Chạy Task Adaptation

```bash
cd WhiteBloodCellClassification
python TA_training.py
```

- Script tự động chạy **Cross-Validation (5 folds)**
- Sau đó chạy **train_final_model** trên toàn bộ dataset

### Bước 5: Benchmark

```bash
# Benchmark checkpoint fold_1 trên test set
python test_fold1.py
```

Kết quả được lưu vào `results/fold1_benchmark_10percent/`:
- `summary.txt` — tổng hợp accuracy
- `*_report.txt` — classification report chi tiết
- `*_correct.png` — 5 ảnh dự đoán đúng
- `*_incorrect.png` — 5 ảnh dự đoán sai

---

## 7. Cấu hình

### Domain Adaptation (`config.yaml`)

| Parameter | Giá trị | Ý nghĩa |
|-----------|---------|---------|
| `LEARNING_RATE` | 0.0002 | AdamW LR cho DA |
| `BATCH_SIZE` | 8 | Batch size (ảnh 256×256) |
| `NUM_EPOCHS` | 30 | Số epoch tối đa |
| `WEIGHT_DECAY` | 0.001 | L2 regularization |
| `SEED` | 140 | Reproducibility |
| `weight_cross_entropy` | 1 | Weight CE loss |
| `weight_mask` | 2 | Weight mask loss |
| `weight_rec` | 2 | Weight reconstruction loss |
| `class_weights` | `[1, 8, 4]` | Background:1, Cytoplasm:8, Nucleus:4 |

### Task Adaptation (`config_task.yaml`)

| Parameter | Giá trị | Ý nghĩa |
|-----------|---------|---------|
| `LEARNING_RATE` | 0.00005 | AdamW LR cho classifier |
| `BATCH_SIZE` | 8 | Batch size |
| `NUM_EPOCHS` | 30 | Số epoch tối đa |
| `NUM_FOLDS` | 5 | Số fold Cross-Validation |
| `PATIENCE` | 5 | Early stopping patience |
| `WEIGHT_DECAY` | 0.001 | L2 regularization |
| `Encoder_lr.weight` | 0.5 | Encoder LR = 0.5 × 0.00005 |
| `SEED` | 2004 | Reproducibility |

---

## Tài liệu tham khảo

- ResNet-50: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- FPN: [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)
- Transformer Decoder: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- PBC Dataset: [Peripheral Blood Cell Dataset for Classification](https://data.mendeley.com/datasets/...)

---

**Lưu ý:** Đây là dự án nghiên cứu. Kết quả trên subset 10% test cho thấy tiềm năng. Để deploy production, nên train final model trên 100% PBC train data và evaluate trên full test set.
