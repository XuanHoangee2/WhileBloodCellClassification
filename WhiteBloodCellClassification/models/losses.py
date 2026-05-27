import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class CLSLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()

        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.class_weights = None
    
    def forward(self, pred_logits, target_classes):
        loss = F.cross_entropy(
            pred_logits,
            target_classes,
            weight=self.class_weights
        )
        return loss
class BCELoss(nn.Module):
    def __init__(self, pos_weight=1.0):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight])

    def forward(self, pred, target):
        return F.binary_cross_entropy_with_logits(
            pred,
            target,
            pos_weight=self.pos_weight.to(pred.device)
        )
###################################################################### 


def get_boundary(x):
    # x: B x K x H x W

    sobel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]], dtype=torch.float32, device=x.device
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]], dtype=torch.float32, device=x.device
    ).view(1, 1, 3, 3)

    B, K, H, W = x.shape
    x = x.reshape(B*K, 1, H, W)

    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)

    grad = torch.sqrt(gx**2 + gy**2 + 1e-6)

    return grad.view(B, K, H, W)

def BoundaryLoss(predictions, masks_long):
    target_onehot = F.one_hot(
        masks_long, num_classes=predictions.shape[1]
    ).permute(0,3,1,2).float()

    pred_prob = torch.softmax(predictions, dim=1)

    pred_edge = get_boundary(pred_prob)
    target_edge = get_boundary(target_onehot)

    loss = F.l1_loss(pred_edge, target_edge)

    return loss
####################################################################
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # pred: (B,1,H,W) logits
        # target: (B,1,H,W) {0,1}

        pred = torch.sigmoid(pred)

        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

        dice = (2 * intersection + self.eps) / (union + self.eps)

        return 1 - dice.mean()
# class BCELoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self,pred_mask, gt_mask):
#         # pred: (B, C, H, W) logits
#         # target: (B, H, W)
        
#         loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask)
#         return loss

# class DiceLoss(nn.Module):
#     def __init__(self, num_classes=3, eps=1e-5):
#         super().__init__()
#         self.num_classes = num_classes
#         self.eps = eps

#     def forward(self, pred, target):
#         # pred: (B, C, H, W) logits
#         # target: (B, H, W)

#         pred = torch.softmax(pred, dim=1)

#         target_onehot = F.one_hot(target, num_classes=self.num_classes)
#         target_onehot = target_onehot.permute(0, 3, 1, 2).float()

#         intersection = (pred * target_onehot).sum(dim=(2,3))
#         union = pred.sum(dim=(2,3)) + target_onehot.sum(dim=(2,3))

#         dice = (2 * intersection + self.eps) / (union + self.eps)

#         return 1 - dice.mean()

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
    def __init__(self, encoder_dim=2048, query_dim=256, use_c3=True):
        super().__init__()
        self.use_c3 = use_c3
        
        self.reconstruction_mlp = nn.Sequential(
            nn.Linear(query_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(512, encoder_dim),
            nn.LayerNorm(encoder_dim)  # Thêm LayerNorm để ổn định
        )
        
        if use_c3:
            self.feature_proj = nn.Linear(1024, encoder_dim)
        else:
            self.feature_proj = None
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.reconstruction_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)  # Giảm gain
                nn.init.constant_(m.bias, 0.0)
        if self.feature_proj is not None:
            nn.init.xavier_uniform_(self.feature_proj.weight, gain=0.5)
            nn.init.constant_(self.feature_proj.bias, 0.0)
    
    def forward(self, encoder_features, query_features):
        # Lấy feature
        if self.use_c3:
            target_feature = encoder_features[2]  # c3: [B, 1024, H, W]
            g_i = target_feature.mean(dim=[2, 3])  # [B, 1024]
            g_i = self.feature_proj(g_i)  # [B, 2048]
        else:
            target_feature = encoder_features[3]  # c4: [B, 2048, H, W]
            g_i = target_feature.mean(dim=[2, 3])  # [B, 2048]
        
        g_i = g_i.detach()
        
        # Lấy query 0
        g_hat = query_features[:, 0, :]  # [B, 256]
        g_hat = self.reconstruction_mlp(g_hat)  # [B, 2048]
        
        loss_raw = F.mse_loss(g_hat, g_i)
    
        cos_sim = F.cosine_similarity(g_hat, g_i, dim=1).mean()
        loss_cos = 1 - cos_sim
        
        g_i_norm = F.normalize(g_i, p=2, dim=1)
        g_hat_norm = F.normalize(g_hat, p=2, dim=1)
        loss_norm_mse = F.mse_loss(g_hat_norm, g_i_norm)
        
        # with torch.no_grad():
        #     print(f"Loss raw (MSE): {loss_raw.item():.6f}")
        #     print(f"Loss cos (1 - cos): {loss_cos.item():.6f}")
        #     print(f"Loss norm MSE: {loss_norm_mse.item():.6f}")
            
        #     # Kiểm tra cosine similarity
        #     cos_sim_value = F.cosine_similarity(g_hat, g_i, dim=1).mean()
        #     print(f"Cosine similarity (raw): {cos_sim_value:.4f}")
            
        #     cos_sim_norm = F.cosine_similarity(g_i_norm, g_hat_norm, dim=1).mean()
        #     print(f"Cosine similarity (norm): {cos_sim_norm:.4f}")
        
        return loss_cos  

# class ReconstructionLoss(nn.Module):
#     def __init__(self, encoder_dim=1024, query_dim=256, num_queries=32):
#         super().__init__()

#         self.reconstruction_mlp = nn.Sequential(
#             nn.Linear(query_dim * num_queries, encoder_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(encoder_dim, encoder_dim)
#         )

#     def forward(self, encoder_features, query_features):


#         g_i = encoder_features.mean(dim=[2, 3])
#         g_i = F.normalize(g_i, p=2, dim=1)


#         g_hat = query_features.flatten(1)
#         g_hat = self.reconstruction_mlp(g_hat)
#         g_hat = F.normalize(g_hat, p=2, dim=1)

#         return F.mse_loss(g_hat, g_i)
class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, y_hat, y):
        return self.loss_fn(y_hat, y)