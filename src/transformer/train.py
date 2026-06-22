from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformer.config import ModelConfig, TrainConfig
from transformer.model.gpt import GPT


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    batch, seq_len, vocab = logits.shape
    return F.cross_entropy(logits.view(batch * seq_len, vocab), targets.view(batch * seq_len))


def train_step(
    model: GPT,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    optimizer.zero_grad()
    logits = model(x)
    loss = compute_loss(logits, y)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate_loss(model: GPT, dataloader: DataLoader, device: str, max_batches: int | None = None) -> float:
    model.eval()
    losses: list[float] = []
    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        losses.append(compute_loss(logits, y).item())
    return sum(losses) / len(losses) if losses else 0.0


def train_one_epoch(
    model: GPT,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    train_config: TrainConfig,
) -> list[float]:
    losses: list[float] = []
    for step, batch in enumerate(dataloader):
        if step >= train_config.max_steps:
            break
        loss = train_step(model, batch, optimizer, train_config.device)
        losses.append(loss)
    return losses


def save_checkpoint(
    path: str | Path,
    model: GPT,
    model_config: ModelConfig,
    tokenizer_dict: dict,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "model_config": model_config.__dict__,
        "tokenizer": tokenizer_dict,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[GPT, ModelConfig, dict, dict | None]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model_config = ModelConfig(**payload["model_config"])
    model = GPT(model_config)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    optimizer_state = payload.get("optimizer_state")
    return model, model_config, payload["tokenizer"], optimizer_state
