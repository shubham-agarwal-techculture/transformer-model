from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformer.config import ModelConfig, TrainConfig
from transformer.model.gpt import GPT


@dataclass
class CheckpointMetadata:
    step: int = 0
    max_steps: int | None = None
    eval_loss: float | None = None


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
    metadata: CheckpointMetadata | None = None,
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
    if metadata is not None:
        payload["step"] = metadata.step
        payload["max_steps"] = metadata.max_steps
        payload["eval_loss"] = metadata.eval_loss
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path, device: str = "cpu"
) -> tuple[GPT, ModelConfig, dict, dict | None, CheckpointMetadata]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model_config = ModelConfig(**payload["model_config"])
    model = GPT(model_config)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    optimizer_state = payload.get("optimizer_state")
    metadata = CheckpointMetadata(
        step=payload.get("step", 0),
        max_steps=payload.get("max_steps"),
        eval_loss=payload.get("eval_loss"),
    )
    return model, model_config, payload["tokenizer"], optimizer_state, metadata


def persist_training_checkpoint(
    out_dir: str | Path,
    step: int,
    model: GPT,
    model_config: ModelConfig,
    tokenizer_dict: dict,
    optimizer: torch.optim.Optimizer,
    *,
    max_steps: int,
    eval_loss: float | None = None,
    numbered: bool = False,
) -> Path:
    """Save latest.pt and optionally a step-numbered checkpoint."""
    out_dir = Path(out_dir)
    metadata = CheckpointMetadata(step=step, max_steps=max_steps, eval_loss=eval_loss)

    latest_path = out_dir / "latest.pt"
    save_checkpoint(latest_path, model, model_config, tokenizer_dict, optimizer, metadata)

    if numbered:
        step_path = out_dir / f"step_{step:06d}.pt"
        save_checkpoint(step_path, model, model_config, tokenizer_dict, optimizer, metadata)

    return latest_path
