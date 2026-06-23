from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from transformer.tokenizer import BPETokenizer


GUTENBERG_START = "*** START OF"
GUTENBERG_END = "*** END OF"


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def prepare_text(text: str, strip_gutenberg: bool = True) -> str:
    """Optionally strip Project Gutenberg header/footer boilerplate."""
    if not strip_gutenberg:
        return text

    start = text.find(GUTENBERG_START)
    if start != -1:
        content_start = text.find("\n", start)
        if content_start != -1:
            text = text[content_start + 1 :]

    end = text.find(GUTENBERG_END)
    if end != -1:
        text = text[:end]

    return text.strip()


class TextChunkDataset(Dataset):
    """Sliding-window dataset where y is x shifted by one token."""

    def __init__(self, text: str, tokenizer: BPETokenizer, block_size: int) -> None:
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
    tokenizer: BPETokenizer,
    block_size: int,
    batch_size: int,
    shuffle: bool = True,
    runtime: "RuntimeConfig | None" = None,
) -> DataLoader:
    from transformer.runtime import RuntimeConfig, dataloader_kwargs

    dataset = TextChunkDataset(text, tokenizer, block_size)
    loader_kwargs = dataloader_kwargs(runtime) if runtime is not None else {}
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **loader_kwargs)
