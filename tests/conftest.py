import pytest
import torch

from transformer.config import ModelConfig, TrainConfig
from transformer.tokenizer import CharTokenizer


@pytest.fixture
def tiny_corpus() -> str:
    return (
        "abcabcabcabcabcabcabcabcabcabc"
        "defghijklmnopqrstuvwxyz0123456789"
        "The quick brown fox jumps over the lazy dog."
    )


@pytest.fixture
def tiny_tokenizer(tiny_corpus: str) -> CharTokenizer:
    return CharTokenizer.from_text(tiny_corpus)


@pytest.fixture
def tiny_config(tiny_tokenizer: CharTokenizer) -> ModelConfig:
    return ModelConfig(
        vocab_size=tiny_tokenizer.vocab_size,
        d_model=32,
        n_heads=4,
        n_layers=2,
        block_size=16,
        dropout=0.0,
    )


@pytest.fixture
def train_config() -> TrainConfig:
    return TrainConfig(lr=1e-3, batch_size=4, max_steps=10, device="cpu")


@pytest.fixture
def device() -> str:
    return "cpu"


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(42)
