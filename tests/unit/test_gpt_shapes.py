import pytest
import torch

from transformer.config import ModelConfig
from transformer.model.gpt import GPT


def test_gpt_logits_shape(tiny_config):
    model = GPT(tiny_config)
    batch, seq_len = 2, tiny_config.block_size
    idx = torch.randint(0, tiny_config.vocab_size, (batch, seq_len))
    logits = model(idx)
    assert logits.shape == (batch, seq_len, tiny_config.vocab_size)


def test_gpt_rejects_long_sequence(tiny_config):
    model = GPT(tiny_config)
    idx = torch.zeros(1, tiny_config.block_size + 1, dtype=torch.long)
    with pytest.raises(ValueError, match="exceeds block_size"):
        model(idx)
