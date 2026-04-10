# White Blood Cell Classification Project

## Overview
This is a **White Blood Cell (WBC) Classification and Segmentation** project that uses a deep learning approach combining **Transformer-based architecture** with **Domain Adaptation** techniques. The system performs semantic segmentation to classify white blood cells into three classes: background, cytoplasm, and nucleus.

## Architecture

### Core Components

#### 1. **PixelEncoder** (`models/PixelEncoder.py`)
- Uses **ResNet-50** as the backbone encoder
- Extracts multi-scale features (c1, c2, c3, c4) from input images
- BatchNorm layers are frozen during training

#### 2. **PixelDecoder** (`models/PixelDecoder.py`)
- FPN-style (Feature Pyramid Network) decoder
- Uses lateral connections and upsampling to fuse multi-scale features
- Produces pixel-level features at 256 channels

#### 3. **Spatial Co-occurrence Feature Extraction (SCFE)** (`models/spatial_cooccurrence.py`)
- Extracts spatial relationships using co-occurrence patterns
- Consists of:
  - MLP projection layer
  - Co-occurrence layer (CoLLayer) with learnable spatial weights
  - Global average pooling to generate query embeddings

#### 4. **TransformerDecoder** (`models/TransformerDecoder.py`)
- **6-layer Transformer decoder** with:
  - Self-attention mechanism
  - Cross-attention between queries and pixel memory
  - Feed-forward networks with LayerNorm
- Uses **32 learnable query embeddings**
- First query is enhanced with SCFE features

#### 5. **SegmentationHead** (`models/segmentationHead.py`)
- Generates final segmentation masks
- Uses mask embeddings and class probabilities
- Implements temperature-scaled softmax (temperature=0.6)
- Outputs 3-class segmentation (background, cytoplasm, nucleus)

### Supporting Modules

#### **Blocks** (`models/blocks.py`)
- `ConvBlock`: Standard conv-bn-relu block
- `MLPLayer`: Multi-layer perceptron with Conv2d
- `CoLLayer`: Co-occurrence layer with cosine similarity-based spatial weighting

#### **Losses** (`models/losses.py`)
- `CLSLoss`: Cross-entropy loss with class weighting
- `DiceLoss`: Dice coefficient loss for segmentation
- `BCELoss`: Binary cross-entropy loss
- `MaskLoss`: Combined BCE + Dice loss
- `BoundaryLoss`: Edge detection loss using Sobel operators
- `ReconstructionLoss`: Feature reconstruction loss with cosine similarity

## Domain Adaptation Module (`DomainAdaptation/DA_module.py`)
Integrates all components into an end-to-end trainable model:
- Encoder → Decoder → SCFE → TransformerDecoder → SegmentationHead
- Returns segmentation masks, encoded features, and query features

## Training Pipeline (`DA_training.py`)

### Training Configuration
- **Optimizer**: AdamW with learning rate 0.0002
- **Batch Size**: 8
- **Epochs**: 30
- **Device**: CUDA (GPU)
- **Mixed Precision**: Enabled via GradScaler

### Loss Function (Combined)
```
Total Loss = 1.0 × CrossEntropy + 2.0 × MaskLoss + 2.0 × ReconstructionLoss
```

### Class Weights
- Background: 1.0
- Cytoplasm: 2.5
- Nucleus: 2.0

## Dataset (`dataset.py`)

### JSTCDataset
- Loads paired images (.bmp) and masks (.png)
- Supports 80/20 train-validation split
- Label mapping:
  - 0: Background
  - 1: Cytoplasm (128 in mask)
  - 2: Nucleus (255 in mask)
- Input size: 256×256

## Utilities

### Config System (`config/config_loader.py`, `config/config.yaml`)
YAML-based configuration for:
- Training hyperparameters
- Dataset paths
- Loss weights
- Class weights
- Model temperature settings

### Logger (`logger.py`)
- CSV logging of training metrics per epoch
- Automatic checkpoint saving every 5 epochs
- Config snapshot with timestamped experiment directories

### Utils (`utils.py`)
- Optimizer setup (AdamW)
- Gradient scaler for mixed precision
- Checkpoint save/load utilities
- Accuracy evaluation

## Project Structure
```
WhiteBloodCellClassification/
├── models/
│   ├── PixelEncoder.py          # ResNet-50 backbone
│   ├── PixelDecoder.py          # FPN decoder
│   ├── TransformerDecoder.py    # Query-based transformer
│   ├── segmentationHead.py      # Mask generation head
│   ├── spatial_cooccurrence.py  # SCFE module
│   ├── blocks.py                # Building blocks
│   ├── losses.py                # Loss functions
│   └── classifier.py            # Classification head
├── DomainAdaptation/
│   └── DA_module.py             # End-to-end DA model
├── config/
│   ├── config.yaml              # Training config
│   └── config_loader.py         # Config loader
├── dataset.py                   # JSTCDataset
├── DA_training.py              # Training script
├── utils.py                     # Utility functions
├── logger.py                    # Experiment logger
└── __init__.py

data/
├── RawData/
│   └── Dataset 1/              # Image and mask data
└── SourceData/

research/                        # Research notebooks
experiments/checkpoints         # Model checkpoints
```

## Dependencies (`requirements.txt`)
- torch==2.1.0
- torchvision==0.16.0
- numpy==1.26.4
- matplotlib
- opencv-python
- tqdm
- pyyaml

## Quick Start

### Training
```python
python WhiteBloodCellClassification/DA_training.py
```

### Inference (Simple Test)
```python
python train.py
```

## Key Features
1. **Query-based Segmentation**: Uses transformer queries to generate class-aware masks
2. **Spatial Co-occurrence**: Captures spatial relationships in cell structures
3. **Domain Adaptation**: Architecture designed for cross-domain generalization
4. **Multi-scale Features**: FPN decoder aggregates features at multiple resolutions
5. **Combined Loss**: Balances classification, segmentation, and reconstruction objectives

## Model Flow
```
Input Image (3×256×256)
    ↓
PixelEncoder (ResNet-50) → [c1, c2, c3, c4]
    ↓
PixelDecoder (FPN) → Pixel Features (256×H×W)
    ↓
SCFE (on c4) → Query Enhancement
    ↓
TransformerDecoder (6 layers) → Refined Queries
    ↓
SegmentationHead → Masks (3×256×256)
    ↓
Output: Background | Cytoplasm | Nucleus
```
