import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    def __init__(
        self,
        d_model=256,
        pixel_dim=256,
        num_classes=3,
        target_size=(256, 256)
    ):
        super().__init__()
        self.mask_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, pixel_dim)
        )

        self.class_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, num_classes)
        )

        self.target_size = target_size

    def forward(self, queries, pixel_feature):

        B, Nq, C = queries.shape
        _, Cp, H, W = pixel_feature.shape

        mask_embed = self.mask_mlp(queries)  # B x Nq x Cp

        candidate_masks = torch.einsum(
            "bnc,bchw->bnhw",
            mask_embed,
            pixel_feature
        )

        candidate_masks = candidate_masks / (Cp ** 0.5)
        class_logits = self.class_mlp(queries)  # B x Nq x K

        temperature = 0.5  
        class_probs = torch.softmax(class_logits / temperature, dim=-1)

        final_masks = torch.einsum(
            "bnk,bnhw->bkhw",
            class_probs,
            candidate_masks
        )

        final_masks = F.interpolate(
            final_masks,
            size=self.target_size,
            mode="bilinear",
            align_corners=False
        )

        return final_masks