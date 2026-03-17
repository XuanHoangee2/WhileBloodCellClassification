import torch
import torch.nn as nn
import torch.nn.functional as F

class CLSLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_logits, target_classes):
        loss = F.cross_entropy(pred_logits, target_classes)
        return loss
    
class BCELoss(nn.Module):
    def __init__(self):
        super().__iinit__()

    def forward(self,pred_mask, gt_mask):
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask)
        return loss

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, pred_mask, gt_mask):
        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = pred_mask.flatten(1)
        gt_mask = gt_mask.flatten(1)

        intersection = (pred_mask * gt_mask).sum(dim=1)
        union = pred_mask.sum(dim=1) + gt_mask.sum(dim=1)
        dice = (2 * intersection + self.eps) / (union + self.eps)
        loss = 1 - dice.mean()
        return loss

class MaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
    
    def forward(self, pred_mask, gt_mask):
        bce_loss = self.bce(pred_mask, gt_mask)
        dice_loss = self.dice(pred_mask, gt_mask)
        loss = bce_loss + dice_loss
        return loss

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, encoder_feat, decoder_feat):
        g_encoder = encoder_feat.mean(dim=[2,3]) #[B, C, H, W] → [B, C]
        g_decoder = decoder_feat.mean(dim=1) #[B, num_queries, C] → [B, C]

        g_encoder = F.normalize(g_encoder, dim=-1)
        g_decoder = F.normalize(g_decoder, dim=-1)

        loss = F.mse_loss(
            g_encoder,
            g_decoder
        )

        return loss