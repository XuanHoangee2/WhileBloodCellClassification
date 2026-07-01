import torch.nn as nn
import torch
import torch.nn.functional as F


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, query, memory, attn_mask=None):
        """
        query: (B, Nq, C)
        memory: (B, HW, C)  - pixel features flattened
        attn_mask: (B*nhead, Nq, HW) additive float mask {0, -inf} hoặc None
                   (None => cross-attention toàn cục, dùng cho layer đầu tiên
                   khi chưa có mask dự đoán nào từ layer trước)
        """
        # self attention giữa các query
        q = self.self_attn(query, query, query)[0]
        query = self.norm1(query + q)

        # masked cross-attention: query chỉ tập trung vào vùng pixel
        # được dự đoán bởi mask của layer trước (Eq. masked attention, Mask2Former)
        c = self.cross_attn(query, memory, memory, attn_mask=attn_mask)[0]
        query = self.norm2(query + c)

        # feed forward
        f = self.ffn(query)
        query = self.norm3(query + f)
        return query


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers=6, d_model=256, nhead=8, num_queries=32):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_queries = num_queries

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, nhead) for _ in range(num_layers)]
        )

        # Mask embedding dùng để dự đoán mask trung gian sau mỗi layer,
        # phục vụ masked cross-attention cho layer kế tiếp.
        self.mask_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

    def predict_mask(self, query, pixel_features, H, W):
        """
        Dự đoán mask logits từ query hiện tại và pixel features.
        query: (B, Nq, C)
        pixel_features: (B, C, H, W)
        return: mask_logits (B, Nq, H, W)
        """
        mask_embed = self.mask_embed(query)  # (B, Nq, C)
        mask_logits = torch.einsum("bnc,bchw->bnhw", mask_embed, pixel_features)
        return mask_logits

    def build_attn_mask(self, mask_logits, B):
        """
        Chuyển mask logits thành additive attention mask {0, -inf}
        theo đúng công thức masked attention của Mask2Former:
            MaskedAttn(Q,K,V) = softmax(QK^T/sqrt(dk) + logM) V
        với M là mask nhị phân (1 nếu pixel thuộc vùng được attend, 0 nếu không).

        mask_logits: (B, Nq, H, W)
        return: attn_mask (B*nhead, Nq, HW) kiểu float, giá trị 0 hoặc -inf
        """
        Nq = mask_logits.shape[1]
        HW = mask_logits.shape[2] * mask_logits.shape[3]

        # Nhị phân hóa: vùng được dự đoán là đối tượng (sigmoid > 0.5) thì attend được
        mask_prob = torch.sigmoid(mask_logits).flatten(2)  # (B, Nq, HW)
        keep = mask_prob > 0.5  # True = được attend

        # Tránh trường hợp một query không attend được pixel nào (toàn bộ bị mask)
        # -> mở toàn bộ attention cho query đó để tránh NaN khi softmax
        empty_query = (~keep).all(dim=-1)  # (B, Nq)
        if empty_query.any():
            keep = keep.clone()
            keep[empty_query] = True

        # additive mask: 0 nếu được attend, -inf nếu bị che
        attn_mask = torch.zeros_like(mask_prob)
        attn_mask = attn_mask.masked_fill(~keep, float("-inf"))

        # Mở rộng cho từng head: (B, Nq, HW) -> (B*nhead, Nq, HW)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1)
        attn_mask = attn_mask.view(B * self.nhead, Nq, HW)
        return attn_mask

    def forward(self, pixel_features, scfe_query):
        B, C, H, W = pixel_features.shape
        memory = pixel_features.flatten(2).permute(0, 2, 1)  # (B, HW, C)

        query = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        query[:, 0, :] = query[:, 0, :] + scfe_query.squeeze(1)

        attn_mask = None  # layer đầu tiên: chưa có mask dự đoán, dùng cross-attn toàn cục

        for layer in self.layers:
            query = layer(query, memory, attn_mask=attn_mask)

            # Dự đoán mask trung gian từ query vừa cập nhật, dùng cho layer kế tiếp
            mask_logits = self.predict_mask(query, pixel_features, H, W)
            attn_mask = self.build_attn_mask(mask_logits, B)

        return query





# import torch.nn as nn 
# import torch 

# class TransformerDecoderLayer(nn.Module): 
#     def __init__(self, d_model = 256, nhead = 8, dim_feedforward = 512, dropout= 0.1): 
#         super().__init__() 
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True) 
#         self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True) 
#         self.ffn = nn.Sequential( nn.Linear(d_model,dim_feedforward), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout) )
#         self.norm1 = nn.LayerNorm(d_model) 
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model) 
#     def forward(self, query, memory): 
#         #self attention
#         q = self.self_attn(query, query, query)[0] 
#         query = self.norm1(query + q) 
#         #cross attention
#         c = self.cross_attn(query, memory, memory)[0] 
#         query = self.norm2(query + c) 
#         #feed forward
#         f = self.ffn(query) 
#         query = self.norm3(query + f) 
#         return query 

# class TransformerDecoder(nn.Module): 
#     def __init__(self, num_layers = 6, d_model = 256, nhead = 8, num_queries = 32):
#         super().__init__() 
#         self.query_embed = nn.Embedding(num_queries, d_model) 
#         self.layers = nn.ModuleList([ TransformerDecoderLayer(d_model, nhead) for _ in range(num_layers) ]) 

#     def forward(self, pixel_features, scfe_query): 
#         B, C, H, W = pixel_features.shape 
#         memory = pixel_features.flatten(2).permute(0,2,1) # B, HW, C 
#         query = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1) 
#         # alpha = nn.Parameter(torch.tensor(0.1))
#         # query = query + alpha * scfe_query
#         query[:, 0, :] = query[:, 0, :] + scfe_query.squeeze(1)
#         for layer in self.layers: 
#             query = layer(query, memory) 

#         return query 