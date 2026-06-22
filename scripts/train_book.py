#!/usr/bin/env python3
"""Train a GPT model on a book or text file."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from transformer.config import ModelConfig, TrainConfig
from transformer.dataset import create_dataloader, load_text
from transformer.model.gpt import GPT
from transformer.tokenizer import CharTokenizer
from transformer.train import evaluate_loss, save_checkpoint, train_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GPT model on text")
    parser.add_argument("--data", type=str, required=True, help="Path to text file")
    parser.add_argument("--out", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--max-steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = load_text(args.data)
    tokenizer = CharTokenizer.from_text(text)

    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        block_size=args.block_size,
    )
    train_config = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        device=args.device,
    )

    dataloader = create_dataloader(
        text, tokenizer, model_config.block_size, train_config.batch_size
    )
    model = GPT(model_config).to(train_config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(range(train_config.max_steps), desc="Training")
    data_iter = iter(dataloader)
    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        loss = train_step(model, batch, optimizer, train_config.device)
        pbar.set_postfix(loss=f"{loss:.4f}")

        if (step + 1) % train_config.eval_interval == 0:
            eval_loss = evaluate_loss(model, dataloader, train_config.device, max_batches=5)
            pbar.write(f"Step {step + 1}: eval loss = {eval_loss:.4f}")

    latest_path = out_dir / "latest.pt"
    save_checkpoint(latest_path, model, model_config, tokenizer.to_dict(), optimizer)
    print(f"Saved checkpoint to {latest_path}")


if __name__ == "__main__":
    main()
