# Transformer Book Trainer

A simple, from-scratch GPT-style decoder-only transformer built with PyTorch. Train it on any book or text file and generate continuations.

## Setup

```bash
pip install -e ".[dev]"
```

## Add training data

Place your text file at `data/book.txt` (or pass any path via `--data`).

## Train

```bash
python scripts/train_book.py --data data/book.txt --out checkpoints/
```

## Generate

```bash
python scripts/generate.py --checkpoint checkpoints/latest.pt --prompt "Chapter 1"
```

## Test

```bash
pytest
```

## Project layout

```
src/transformer/     Core library (config, tokenizer, model, train, generate)
scripts/             CLI entrypoints
tests/               unit, integration, acceptance, regression tests
data/                Place your book text here
```
