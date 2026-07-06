import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query, key, value):
        out, _ = self.mha(query, key, value)
        return out