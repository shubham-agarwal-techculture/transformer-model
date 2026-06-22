import torch.nn as nn

from transformer.model.attention import CausalSelfAttention
from transformer.model.ffn import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
