import torch.nn as nn 
import torch 

class TransformerDecoderLayer(nn.Module): 
    def __init__(self, d_model = 256, nhead = 8, dim_feedforward = 512, dropout= 0.1): 
        super().__init__() 
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True) 
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True) 
        self.ffn = nn.Sequential( nn.Linear(d_model,dim_feedforward), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout) )
        self.norm1 = nn.LayerNorm(d_model) 
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model) 
    def forward(self, query, memory): 
        #self attention
        q = self.self_attn(query, query, query)[0] 
        query = self.norm1(query + q) 
        #cross attention
        c = self.cross_attn(query, memory, memory)[0] 
        query = self.norm2(query + c) 
        #feed forward
        f = self.ffn(query) 
        query = self.norm3(query + f) 
        return query 

class TransformerDecoder(nn.Module): 
    def __init__(self, num_layers = 6, d_model = 256, nhead = 8, num_queries = 32):
        super().__init__() 
        self.query_embed = nn.Embedding(num_queries, d_model) 
        self.layers = nn.ModuleList([ TransformerDecoderLayer(d_model, nhead) for _ in range(num_layers) ]) 

    def forward(self, pixel_features, scfe_query): 
        B, C, H, W = pixel_features.shape 
        memory = pixel_features.flatten(2).permute(0,2,1) # B, HW, C 
        query = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1) 
        # alpha = nn.Parameter(torch.tensor(0.1))
        # query = query + alpha * scfe_query
        query[:, 0, :] = query[:, 0, :] + scfe_query.squeeze(1)
        for layer in self.layers: 
            query = layer(query, memory) 

        return query 