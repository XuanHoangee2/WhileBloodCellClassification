import torch
import torch.nn as nn
import torch.nn.functional as F

class NuclearCytoplasmicClassifier(nn.Module):
    def __init__(self, in_channels=2048, num_classes=8, 
                 hidden_dim=512, dropout=0.5):
        super().__init__()
        
        # Conv 1x1 để transform đặc trưng encoder
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # MLP classifier - input là concatenation của 3 đặc trưng
        # 3 * in_channels = 6144 chiều
        self.mlp = nn.Sequential(
            nn.Linear(3 * in_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
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

    def forward(self, f_j, z_hat):
        """
        Args:
            f_j (Tensor): Encoder features (B, C, H_f, W_f) 
                         e.g., (batch, 2048, 8, 8)
            z_hat (Tensor): Segmentation map (B, K, H_z, W_z)
                           CHỈ LẤY 2 LỚP: nucleus và cytoplasm
                           e.g., (batch, 2, 224, 224)
                           - z_hat[:, 0, :, :] = nucleus probability
                           - z_hat[:, 1, :, :] = cytoplasm probability
        Returns:
            logits (Tensor): Classification logits (B, num_classes)
        """
        B, C, H_f, W_f = f_j.shape
        _, K, H_z, W_z = z_hat.shape
        
        # Kiểm tra: CHỈ dùng 2 lớp (nucleus + cytoplasm)
        assert K == 2, f"Expected 2 seg classes (nucleus, cytoplasm), got {K}"
        
        # Step 1: Transform encoder features
        f_conv = self.conv(f_j)  # (B, C, H_f, W_f)
        
        # Step 2: Downsample segmentation map về resolution của f_j
        z_down = F.interpolate(
            z_hat,
            size=(H_f, W_f),
            mode='bilinear',
            align_corners=False
        )  # (B, K, H_f, W_f)
        
        # Step 3: Convert to probabilities (sigmoid cho phép overlap)
        z_prob = torch.sigmoid(z_down)  # (B, K, H_f, W_f)
        
        # Step 4: Trích xuất đặc trưng riêng cho từng vùng
        # (B, C, H_f, W_f) * (B, 1, H_f, W_f) -> (B, C, H_f, W_f)
        f_nucleus = f_conv * z_prob[:, 0:1, :, :]   # Chỉ vùng nhân
        f_cytoplasm = f_conv * z_prob[:, 1:2, :, :] # Chỉ vùng bào tương
        f_cell = f_conv * (z_prob[:, 0:1, :, :] + z_prob[:, 1:2, :, :])  # Toàn bộ tế bào
        
        # Step 5: Global Average Pooling cho từng vùng
        f_nucleus_pool = f_nucleus.sum(dim=(2, 3)) / (z_prob[:, 0:1, :, :].sum(dim=(2, 3)) + 1e-6)
        f_cytoplasm_pool = f_cytoplasm.sum(dim=(2, 3)) / (z_prob[:, 1:2, :, :].sum(dim=(2, 3)) + 1e-6)
        f_cell_pool = f_cell.sum(dim=(2, 3)) / ((z_prob[:, 0:1, :, :] + z_prob[:, 1:2, :, :]).sum(dim=(2, 3)) + 1e-6)
        
        # Step 6: CONCATENATE 3 đặc trưng (như luận văn)
        f_final = torch.cat([f_nucleus_pool, f_cytoplasm_pool, f_cell_pool], dim=1)
        # (B, 3*C) = (B, 6144)
        
        # Step 7: MLP classifier
        logits = self.mlp(f_final)  # (B, num_classes)
        
        return logits