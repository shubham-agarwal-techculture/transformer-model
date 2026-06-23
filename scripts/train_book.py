#!/usr/bin/env python3
"""Train a GPT model on a book or text file."""

from __future__ import annotations

import argparse
import math
import signal
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from transformer.config import ModelConfig, TrainConfig
from transformer.dataset import create_dataloader, load_text, prepare_text
from transformer.model.gpt import GPT
from transformer.runtime import (
    apply_runtime_config,
    build_runtime_config,
    format_runtime_summary,
    probe_hardware,
    suggest_batch_size,
)
from transformer.tokenizer import CharTokenizer
from transformer.train import (
    evaluate_loss,
    load_checkpoint,
    persist_training_checkpoint,
    prepare_model_for_training,
    train_step,
)


def default_device() -> str:
    return "auto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GPT model on text")
    parser.add_argument("--data", type=str, required=True, help="Path to text file")
    parser.add_argument("--out", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a checkpoint path (e.g. checkpoints/latest.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=3.0,
        help="Number of full passes over the text (ignored if --max-steps is set)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Fixed training steps; overrides --epochs when set",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="Save latest.pt and a numbered checkpoint every N steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: auto from hardware profile)",
    )
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--device",
        type=str,
        default=default_device(),
        help="Device: auto, cpu, cuda, or mps",
    )
    parser.add_argument(
        "--runtime-preset",
        choices=["auto", "minimal"],
        default="auto",
        help="auto=tune for detected hardware; minimal=portable baseline",
    )
    parser.add_argument(
        "--runtime-config",
        type=str,
        default=None,
        help="Optional JSON file overriding RuntimeConfig fields",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker processes (default: auto from CPU count)",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use torch.compile when supported",
    )
    parser.add_argument(
        "--strip-gutenberg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove Project Gutenberg header/footer when present",
    )
    return parser.parse_args()


class TrainingSession:
    def __init__(
        self,
        model: GPT,
        model_config: ModelConfig,
        tokenizer: CharTokenizer,
        optimizer: torch.optim.Optimizer,
        dataloader,
        train_config: TrainConfig,
        runtime,
        out_dir: Path,
        max_steps: int,
        save_every: int,
        start_step: int = 0,
    ) -> None:
        self.model = model
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.train_config = train_config
        self.runtime = runtime
        self.out_dir = out_dir
        self.max_steps = max_steps
        self.save_every = save_every
        self.current_step = start_step

    def save(self, reason: str, *, numbered: bool = True) -> Path:
        eval_loss = evaluate_loss(
            self.model,
            self.dataloader,
            self.train_config.device,
            max_batches=5,
            runtime=self.runtime,
        )
        path = persist_training_checkpoint(
            self.out_dir,
            self.current_step,
            self.model,
            self.model_config,
            self.tokenizer.to_dict(),
            self.optimizer,
            max_steps=self.max_steps,
            eval_loss=eval_loss,
            numbered=numbered,
        )
        print(
            f"Checkpoint saved at step {self.current_step}/{self.max_steps} "
            f"(eval loss={eval_loss:.4f}, reason={reason}) -> {path}"
        )
        return path

    def run(self) -> None:
        pbar = tqdm(range(self.current_step, self.max_steps), desc="Training", initial=self.current_step, total=self.max_steps)
        data_iter = iter(self.dataloader)

        try:
            for step in pbar:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)

                loss = train_step(
                    self.model,
                    batch,
                    self.optimizer,
                    self.train_config.device,
                    runtime=self.runtime,
                )
                self.current_step = step + 1
                pbar.set_postfix(loss=f"{loss:.4f}")

                if self.current_step % self.train_config.eval_interval == 0:
                    eval_loss = evaluate_loss(
                        self.model,
                        self.dataloader,
                        self.train_config.device,
                        max_batches=5,
                        runtime=self.runtime,
                    )
                    pbar.write(f"Step {self.current_step}: eval loss = {eval_loss:.4f}")

                if self.current_step % self.save_every == 0:
                    self.save("periodic")

            self.save("complete")
        except KeyboardInterrupt:
            self.save("interrupted")
            print("\nTraining interrupted. Resume with:")
            print(f"  python scripts/train_book.py --data <text> --resume {self.out_dir / 'latest.pt'}")
            sys.exit(130)


def _install_signal_handlers(session: TrainingSession) -> None:
    def _handle_signal(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)


def main() -> None:
    args = parse_args()
    profile = probe_hardware()
    runtime = build_runtime_config(
        profile=profile,
        preset=args.runtime_preset,
        device=args.device,
        config_path=args.runtime_config,
        compile_model=args.compile,
        num_workers=args.num_workers,
    )
    apply_runtime_config(runtime)

    raw_text = load_text(args.data)
    text = prepare_text(raw_text, strip_gutenberg=args.strip_gutenberg)
    tokenizer = CharTokenizer.from_text(text)

    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        block_size=args.block_size,
    )
    batch_size = suggest_batch_size(runtime, model_config, requested=args.batch_size)
    train_config = TrainConfig(
        lr=args.lr,
        batch_size=batch_size,
        max_steps=0,
        device=runtime.device,
    )

    dataloader = create_dataloader(
        text,
        tokenizer,
        model_config.block_size,
        train_config.batch_size,
        runtime=runtime,
    )
    steps_per_epoch = math.ceil(len(dataloader.dataset) / train_config.batch_size)
    if args.max_steps is not None:
        max_steps = args.max_steps
    else:
        max_steps = max(1, int(args.epochs * steps_per_epoch))
    train_config.max_steps = max_steps

    start_step = 0
    optimizer_state = None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        model, resumed_config, tokenizer_dict, optimizer_state, metadata = load_checkpoint(
            args.resume, device=train_config.device
        )
        if resumed_config.vocab_size != model_config.vocab_size:
            raise ValueError(
                "Checkpoint vocab size does not match training text. "
                "Use the same text file or train from scratch."
            )
        model_config = resumed_config
        tokenizer = CharTokenizer.from_dict(tokenizer_dict)
        start_step = metadata.step
        if args.max_steps is None and metadata.max_steps is not None:
            max_steps = metadata.max_steps
            train_config.max_steps = max_steps
        print(f"Resuming from step {start_step} (checkpoint: {args.resume})")
    else:
        model = GPT(model_config)

    model = prepare_model_for_training(model, runtime)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    print(f"Text length: {len(text):,} characters")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Steps per epoch: {steps_per_epoch:,}")
    print(f"Training for {max_steps:,} steps (~{max_steps / steps_per_epoch:.2f} epochs)")
    print(f"Batch size: {train_config.batch_size}")
    print(f"Saving every {args.save_every:,} steps to {out_dir}")
    print(format_runtime_summary(runtime, profile))

    session = TrainingSession(
        model=model,
        model_config=model_config,
        tokenizer=tokenizer,
        optimizer=optimizer,
        dataloader=dataloader,
        train_config=train_config,
        runtime=runtime,
        out_dir=out_dir,
        max_steps=max_steps,
        save_every=args.save_every,
        start_step=start_step,
    )
    _install_signal_handlers(session)
    session.run()


if __name__ == "__main__":
    main()
