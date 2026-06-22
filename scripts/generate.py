#!/usr/bin/env python3
"""Generate text from a trained checkpoint."""

from __future__ import annotations

import argparse

from transformer.generate import generate
from transformer.tokenizer import CharTokenizer
from transformer.train import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding (temperature=0)")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, _, tokenizer_dict, _, _ = load_checkpoint(args.checkpoint, device=args.device)
    tokenizer = CharTokenizer.from_dict(tokenizer_dict)

    temperature = 0.0 if args.greedy else args.temperature
    output = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        temperature=temperature,
        top_k=None if args.greedy else args.top_k,
    )
    print(output)


if __name__ == "__main__":
    main()
