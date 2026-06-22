from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from transformer.tokenizer import CharTokenizer


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


class TextChunkDataset(Dataset):
    """Sliding-window dataset where y is x shifted by one token."""

    def __init__(self, text: str, tokenizer: CharTokenizer, block_size: int) -> None:
        self.block_size = block_size
        self.tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        if len(self.tokens) <= block_size:
            raise ValueError("Text must be longer than block_size")

    def __len__(self) -> int:
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def create_dataloader(
    text: str,
    tokenizer: CharTokenizer,
    block_size: int,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = TextChunkDataset(text, tokenizer, block_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
