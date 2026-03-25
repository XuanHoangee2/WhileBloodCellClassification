import torch
import torch.nn as nn
import torch.nn.functional as F

class CLSLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_logits, target_classes):
        # pred: (B, C, H, W) logits
        # target: (B, H, W)

        loss = F.cross_entropy(pred_logits, target_classes)
        return loss
    
class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,pred_mask, gt_mask):
        # pred: (B, C, H, W) logits
        # target: (B, H, W)
        
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask)
        return loss

class DiceLoss(nn.Module):
    def __init__(self, num_classes=3, eps=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, pred, target):
        # pred: (B, C, H, W) logits
        # target: (B, H, W)

        pred = torch.softmax(pred, dim=1)

        target_onehot = F.one_hot(target, num_classes=self.num_classes)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()

        intersection = (pred * target_onehot).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target_onehot.sum(dim=(2,3))

        dice = (2 * intersection + self.eps) / (union + self.eps)

        return 1 - dice.mean()

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
    """
    Reconstruction Loss theo công thức (6) trong bài báo
    """
    def __init__(self, encoder_dim=2048, query_dim=256):
        super().__init__()
        # MLP để map query_features về cùng dimension với encoder_features
        self.reconstruction_mlp = nn.Sequential(
            nn.Linear(query_dim, query_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(query_dim * 2, encoder_dim)
        )
        self.encoder_dim = encoder_dim
    
    def forward(self, encoder_features, query_features):
        """
        Args:
            encoder_features: f_i [B, C_F, H, W] (ví dụ: [B, 2048, 8, 8])
            query_features: Q_i [B, Nq, d_model] (ví dụ: [B, 100, 256])
        """
        B = encoder_features.shape[0]
        
        # Bước 1: Global Average Pooling trên encoder features (công thức 5)
        # g_i = FL(GAP(f_i))
        g_i = encoder_features.mean(dim=[2, 3])  # [B, C_F] = [B, 2048]
        g_i = F.normalize(g_i, p=2, dim=1)       # L2 normalization
        
        # Bước 2: Xử lý query_features
        # Lấy trung bình các queries (hoặc có thể dùng MLP trên từng query)
        g_hat = query_features.mean(dim=1)  # [B, d_model] = [B, 256]
        
        # Bước 3: MLP để map về cùng dimension với g_i
        g_hat = self.reconstruction_mlp(g_hat)  # [B, C_F] = [B, 2048]
        g_hat = F.normalize(g_hat, p=2, dim=1)  # L2 normalization
        
        # Bước 4: MSE loss (công thức 6)
        rec_loss = F.mse_loss(g_hat, g_i)
        
        return rec_loss