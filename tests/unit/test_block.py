import torch

from transformer.model.block import TransformerBlock
from transformer.model.ffn import FeedForward


def test_ffn_output_shape():
    ffn = FeedForward(d_model=32, dropout=0.0)
    x = torch.randn(2, 8, 32)
    assert ffn(x).shape == (2, 8, 32)


def test_block_output_shape():
    block = TransformerBlock(d_model=32, n_heads=4, block_size=16, dropout=0.0)
    x = torch.randn(2, 8, 32)
    assert block(x).shape == (2, 8, 32)
