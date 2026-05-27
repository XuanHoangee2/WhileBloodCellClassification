import torch
import torch.nn as nn
import torch.nn.functional as F

class NuclearCytoplasmicClassifier(nn.Module):
    """
    Nuclear-cytoplasmic knowledge-aware classifier with DOWNSAMPLING strategy.
    
    This version downsamples segmentation maps to match encoder feature resolution,
    which is memory-efficient and fast while still preserving region-level information.
    
    Args:
        in_channels (int): Encoder feature channels (default: 2048 for ResNet c4)
        seg_classes (int): Number of segmentation classes (typically 2: nucleus, cytoplasm)
        num_classes (int): Number of WBC types to classify
        hidden_dim (int): MLP hidden dimension
        dropout (float): Dropout rate for regularization
        use_sigmoid (bool): Use sigmoid instead of softmax (recommended for overlapping regions)
    """
    def __init__(self, in_channels=2048, seg_classes=3, num_classes=6, 
                 hidden_dim=256, dropout=0.5, use_sigmoid=True):
        super().__init__()
        self.seg_classes = seg_classes
        self.use_sigmoid = use_sigmoid
        
        # Conv 1x1 to transform encoder features
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # MLP classifier
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, f_j, z_hat, z_is_logits=True):
        """
        Args:
            f_j (Tensor): Encoder features (B, C, H_f, W_f) - e.g., (4, 2048, 8, 8)
            z_hat (Tensor): Segmentation map (B, K, H_z, W_z) - e.g., (4, 2, 224, 224)
            z_is_logits (bool): If True, apply sigmoid/softmax to z_hat
            
        Returns:
            logits (Tensor): Classification logits (B, num_classes)
        """
        B, C, H_f, W_f = f_j.shape
        _, K, H_z, W_z = z_hat.shape
        
        assert K == self.seg_classes, \
            f"Expected {self.seg_classes} seg classes, got {K}"
        
        # Step 1: Transform encoder features (keep original size)
        f_conv = self.conv(f_j)  # (B, C, H_f, W_f)
        
        # Step 2: DOWNSAMPLE z_hat to match f_j resolution
        z_down = F.interpolate(
            z_hat,
            size=(H_f, W_f),
            mode='bilinear',
            align_corners=False
        )  # (B, K, H_f, W_f)
        
        # Step 3: Convert segmentation logits to probabilities
        if z_is_logits:
            if self.use_sigmoid:
                # Sigmoid: each class independent (allows nucleus & cytoplasm overlap)
                z_prob = torch.sigmoid(z_down)  # (B, K, H_f, W_f)
            else:
                # Softmax: classes are mutually exclusive
                z_prob = torch.softmax(z_down, dim=1)  # (B, K, H_f, W_f)
        else:
            z_prob = z_down
        
        # Step 4: Weighted pooling - combine features with segmentation weights
        # (B, C, H_f, W_f) * (B, K, H_f, W_f) -> sum over spatial -> (B, K, C)
        class_features = torch.einsum('bchw,bkhw->bkc', f_conv, z_prob)
        
        # Step 5: Aggregate across classes
        aggregated = class_features.sum(dim=1)  # (B, C)
        
        # Step 6: Normalize by total weight to get weighted average
        total_weight = z_prob.sum(dim=(1, 2, 3)).unsqueeze(1)  # (B, 1)
        aggregated = aggregated / (total_weight + 1e-6)  # (B, C)
        
        # Step 7: Classify
        logits = self.mlp(aggregated)  # (B, num_classes)
        
        return logits