import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationHead(nn.Module):

    def __init__(self, d_model, pixel_dim, num_classes=3):

        super().__init__()
        self.mask_mlp = nn.Linear(d_model, pixel_dim)
        self.class_mlp = nn.Linear(d_model, num_classes)

    def forward(self, queries, pixel_feature):

        class_logits = self.class_mlp(queries)
        mask_embed = self.mask_mlp(queries)
        masks = torch.einsum("bqc,bchw->bqhw", mask_embed, pixel_feature)
        masks = F.interpolate(masks, size=(256, 256), mode="bilinear", align_corners=False)

        return class_logits, masks