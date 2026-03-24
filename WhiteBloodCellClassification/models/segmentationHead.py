import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationHead(nn.Module):
    def __init__(self, d_model, pixel_dim, encoder_dim, num_seg_classes=3, num_wbc_classes=8):
        """
        Args:
            d_model: chiều của queries (256)
            pixel_dim: số kênh của pixel feature (C_E, thường 256)
            encoder_dim: số kênh của encoder feature (C_F, thường 2048)
            num_seg_classes: số class segmentation
            num_wbc_classes: số class WBC cần phân loại
        """
        super().__init__()
        # MLP cho mask embedding
        self.mask_mlp = nn.Linear(d_model, pixel_dim)
        self.class_mlp = nn.Linear(d_model, num_seg_classes)
        
        # Conv để biến đổi encoder feature (phương trình: Conv(f_j))
        self.encoder_conv = nn.Conv2d(encoder_dim, pixel_dim, kernel_size=1)
        
        # MLP cho classification (sau GAP)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # GAP
            nn.Flatten(),
            nn.Linear(pixel_dim, pixel_dim // 2),
            nn.GELU(),
            nn.Linear(pixel_dim // 2, num_wbc_classes)
        )
        
        # Upsampling target size
        self.target_size = (256, 256)
    
    def forward(self, queries, pixel_feature, encoder_feature):

        B, Nq, _ = queries.shape
        _, C, H, W = pixel_feature.shape

        # 1. Mask embedding
        mask_embed = self.mask_mlp(queries)
        masks = torch.einsum("bqc,bchw->bqhw", mask_embed, pixel_feature)

        # scale
        masks = masks / (C ** 0.5)

        # 2. Class logits
        class_logits = self.class_mlp(queries)

        # soft grouping (improved)
        weights = torch.softmax(class_logits / 0.01, dim=-1)
        masks = torch.einsum("bqk,bqhw->bkhw", weights, masks)

        # 3. Encoder feature
        encoder_feature = F.interpolate(
            encoder_feature, size=(H,W),
            mode="bilinear", align_corners=False
        )

        encoder_feature = self.encoder_conv(encoder_feature)

        # normalize masks
        # masks = torch.softmax(masks, dim=1)

        # 4. Feature aggregation
        f_hat = (masks.unsqueeze(2) * encoder_feature.unsqueeze(1)).sum(dim=1)

        # 5. Classification
        logits = self.classifier(f_hat)

        # 6. Upsample
        masks = F.interpolate(
            masks,
            size=self.target_size,
            mode="bilinear",
            align_corners=False
        )

        return logits, masks
