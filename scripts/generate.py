#!/usr/bin/env python3
"""Generate text from a trained checkpoint."""

from __future__ import annotations

import argparse

from transformer.generate import generate
from transformer.runtime import (
    apply_runtime_config,
    build_runtime_config,
    format_runtime_summary,
    maybe_compile_model,
    probe_hardware,
)
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
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, or mps")
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
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use torch.compile when supported",
    )
    parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Disable KV-cache inference (slower, useful for debugging)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = probe_hardware()
    runtime = build_runtime_config(
        profile=profile,
        preset=args.runtime_preset,
        device=args.device,
        config_path=args.runtime_config,
        compile_model=args.compile,
    )
    if args.no_kv_cache:
        runtime.inference_use_kv_cache = False
    apply_runtime_config(runtime)

    model, _, tokenizer_dict, _, _ = load_checkpoint(args.checkpoint, device=runtime.device)
    model = maybe_compile_model(model, runtime)
    tokenizer = CharTokenizer.from_dict(tokenizer_dict)

    print(format_runtime_summary(runtime, profile))

    temperature = 0.0 if args.greedy else args.temperature
    output = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=runtime.device,
        temperature=temperature,
        top_k=None if args.greedy else args.top_k,
        use_kv_cache=runtime.inference_use_kv_cache,
    )
    print(output)


if __name__ == "__main__":
    main()
