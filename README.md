# Transformer Book Trainer

A simple, from-scratch **decoder-only GPT-style transformer** built with PyTorch. Train it on any book or plain-text file, then generate continuations character by character.

The project is organized for **progressive disclosure of complexity**: each module does one job, and the full model is assembled from small, readable pieces.

---

## Table of contents

- [How it works](#how-it-works)
- [Architecture overview](#architecture-overview)
- [Training pipeline](#training-pipeline)
- [Generation pipeline](#generation-pipeline)
- [Model internals](#model-internals)
- [Data pipeline](#data-pipeline)
- [Quick start](#quick-start)
- [Runtime performance](#runtime-performance)
- [CLI reference](#cli-reference)
- [Configuration](#configuration)
- [Project layout](#project-layout)
- [Testing](#testing)
- [Tips for better output](#tips-for-better-output)
- [Design principles](#design-principles)
- [License](#license)

---

## How it works

At a high level, the system learns to **predict the next character** given all previous characters in a fixed-size window.

```
  Input text:  "Call me Ishmael"
  Tokenized:   [C, a, l, l,   m, e, ...]
  Training:    given "Call me Ish"  → predict "m"
               given "all me Ishm"  → predict "a"
               ... (every position in every chunk)
  Generation:  start with a prompt, repeatedly predict one character at a time
```

This is **causal language modeling** — the same approach used by GPT models, but with a character-level vocabulary instead of subwords.

---

## Architecture overview

```mermaid
flowchart TB
    subgraph dataLayer [Data layer]
        BookFile[Text file e.g. book.txt]
        Prepare[prepare_text]
        Tokenizer[CharTokenizer]
        Dataset[TextChunkDataset]
        DataLoader[DataLoader]
    end

    subgraph modelLayer [Model layer]
        TokenEmb[Token embedding]
        PosEmb[Position embedding]
        Blocks[Transformer blocks x N]
        LNF[Final LayerNorm]
        LMHead[Language model head]
    end

    subgraph perfLayer [Runtime layer]
        Probe[probe_hardware]
        RuntimeCfg[RuntimeConfig]
        Apply[apply_runtime_config]
    end

    subgraph runtime [Runtime]
        TrainLoop[Training loop]
        Checkpoint[Checkpoint .pt]
        Generate[Text generation]
    end

    Probe --> RuntimeCfg --> Apply
    Apply --> DataLoader
    Apply --> TrainLoop
    BookFile --> Prepare --> Tokenizer --> Dataset --> DataLoader
    DataLoader --> TrainLoop
    TrainLoop --> Blocks
    TokenEmb --> Blocks
    PosEmb --> Blocks
    Blocks --> LNF --> LMHead
    TrainLoop --> Checkpoint
    Checkpoint --> Generate
    LMHead --> Generate
    Tokenizer --> Generate
```

### End-to-end data flow

```mermaid
sequenceDiagram
    participant User
    participant TrainScript as train_book.py
    participant Data as dataset.py
    participant Model as GPT
    participant Disk as checkpoints/

    User->>TrainScript: --data book.txt
    TrainScript->>Data: load_text + prepare_text
    Data->>Data: build CharTokenizer vocab
    Data->>Data: sliding windows x y
    loop each training step
        TrainScript->>Model: forward x
        Model-->>TrainScript: logits
        TrainScript->>TrainScript: cross-entropy loss + backward
    end
    TrainScript->>Disk: save latest.pt
```

---

## Training pipeline

```mermaid
flowchart TD
    A[Load raw text] --> B[Strip Gutenberg boilerplate]
    B --> C[Build char vocabulary]
    C --> R[Probe hardware + build RuntimeConfig]
    R --> D[Training loop]
    D --> E[Forward pass + loss + AdamW step]
    E --> F{Step divisible by save_every?}
    F -->|yes| G[Save latest.pt + step_N.pt]
    F -->|no| H{More steps?}
    G --> H
    H -->|yes| D
    H -->|no| I[Save final checkpoint]
    E --> J{Ctrl+C or SIGTERM?}
    J -->|yes| K[Save latest.pt + step_N.pt then exit]
```

### What happens in one training step

1. Sample a batch of `(x, y)` pairs from the text.
   - `x` = 256 characters (input)
   - `y` = same sequence shifted by 1 (target / next character)
2. Run `x` through the GPT model → logits of shape `(batch, 256, vocab_size)`.
3. Compute cross-entropy between logits and `y`.
4. Backpropagate and update weights with AdamW.

### Sliding window example

For text `"hello"` with `block_size=4`:

```
Position:  0   1   2   3   4
Text:      h   e   l   l   o

Sample 0:
  x = [h, e, l, l]
  y = [e, l, l, o]
```

Each sample teaches the model to predict the next character at every position in the chunk.

---

## Generation pipeline

```mermaid
flowchart TD
    Prompt[User prompt] --> Encode[CharTokenizer.encode]
    Encode --> Context[Token ID list]
    Context --> Prime[Prime KV cache with prompt]
    Prime --> Loop{Generate max_new_tokens}

    Loop --> Forward[Single-token forward with KV cache]
    Forward --> Logits[Logits at last position]
    Logits --> Sample[top-k + temperature sampling]
    Sample --> Append[Append token to context]
    Append --> Trim{Context longer than block_size?}
    Trim -->|yes| RePrime[Re-prime cache from trimmed context]
    Trim -->|no| Loop
    RePrime --> Loop

    Loop -->|done| Decode[CharTokenizer.decode]
    Decode --> Output[Generated text]
```

By default, generation uses a **KV cache** so each new token reuses past attention state instead of re-running the full context. Disable with `--no-kv-cache` when debugging.

### Sampling options

| Mode | Flags | Behavior |
|------|-------|----------|
| Stochastic (default) | `--temperature 0.7 --top-k 40` | Sample from top 40 likely chars, scaled by temperature |
| Greedy | `--greedy` | Always pick the highest-probability character |
| Custom | `--temperature 1.0 --top-k 80` | More random, wider candidate pool |

Lower temperature → more conservative, repetitive output.  
Higher temperature → more creative, but less coherent if the model is under-trained.

---

## Model internals

The GPT model is a stack of identical **transformer blocks** with **causal (masked) self-attention**, so each character can only attend to itself and characters before it — never the future.

### GPT block diagram

```mermaid
flowchart TB
    Input[Token IDs batch x seq]

    Input --> TE[Token Embedding]
    Input --> PE[Position Embedding]
    TE --> Add1((+))
    PE --> Add1
    Add1 --> Drop1[Dropout]

    Drop1 --> Block1[Transformer Block 1]
    Block1 --> Block2[Transformer Block 2]
    Block2 --> BlockDots[...]
    BlockDots --> BlockN[Transformer Block N]

    BlockN --> LNFinal[LayerNorm]
    LNFinal --> LMHead[Linear LM head]
    LMHead --> Logits[Logits batch x seq x vocab]

    subgraph oneBlock [One Transformer Block pre-norm]
        direction TB
        XB[Input x] --> LN1[LayerNorm]
        LN1 --> Attn[Causal Multi-Head Attention]
        Attn --> Res1((+))
        XB --> Res1
        Res1 --> LN2[LayerNorm]
        LN2 --> FFN[Feed-Forward GELU]
        FFN --> Res2((+))
        Res1 --> Res2
        Res2 --> OutBlock[Output]
    end
```

### Causal attention mask

The attention matrix is masked so position `i` cannot look at positions `j > i`:

```
        t0  t1  t2  t3
  t0  [ 1   0   0   0 ]
  t1  [ 1   1   0   0 ]
  t2  [ 1   1   1   0 ]
  t3  [ 1   1   1   1 ]
```

This is what makes the model suitable for **autoregressive generation** — it never cheats by peeking at future characters during training.

### Default model hyperparameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `d_model` | 128 | Embedding / hidden dimension |
| `n_heads` | 4 | Attention heads (`head_dim = 32`) |
| `n_layers` | 4 | Stacked transformer blocks |
| `block_size` | 256 | Max context length (characters) |
| `dropout` | 0.1 | Regularization during training |
| `vocab_size` | auto | Number of unique characters in the text |

**Weight tying:** the token embedding matrix and the output (`lm_head`) layer share the same weights — a standard GPT trick that reduces parameters and improves learning.

---

## Data pipeline

```mermaid
flowchart LR
    Raw[Raw .txt file]
    Strip[prepare_text]
    Clean[Clean book text]
    Vocab[Unique chars sorted]
    Encode[encode to IDs]
    Windows[Sliding windows]
    Batches[PyTorch DataLoader]

    Raw --> Strip
    Strip --> Clean --> Vocab --> Encode --> Windows --> Batches
```

### Character tokenizer

- Vocabulary = every unique character in the training text (letters, digits, punctuation, whitespace, newlines).
- `encode("hello")` → list of integer IDs
- `decode([...])` → original string (lossless round-trip)

### Gutenberg stripping

If the file contains Project Gutenberg markers, boilerplate is removed automatically:

```
*** START OF THE PROJECT GUTENBERG EBOOK ... ***
  → keep only text between START and END markers
*** END OF THE PROJECT GUTENBERG EBOOK ... ***
```

Disable with `--no-strip-gutenberg` if you want to train on the full raw file.

---

## Quick start

### 1. Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+ and PyTorch 2.0+.

### 2. Add training data

Place your text at `data/book.txt`, or pass any path via `--data`.

### 3. Train

Full training (default **3 epochs**, auto-tuned for your hardware):

```bash
python scripts/train_book.py --data data/book.txt --out checkpoints/
```

On startup the script prints a **runtime summary** (CPU count, RAM, thread count, DataLoader workers, suggested batch size). Training also logs text length, vocab size, steps per epoch, and eval loss every 100 steps.

Quick trial on CPU:

```bash
python scripts/train_book.py --data data/book.txt --max-steps 5000
```

Use a JSON runtime profile for explicit control:

```bash
python scripts/train_book.py --data data/book.txt --runtime-config configs/runtime.example.json
```

### 4. Generate

```bash
python scripts/generate.py --checkpoint checkpoints/latest.pt --prompt "Call me Ishmael."
python scripts/generate.py --checkpoint checkpoints/latest.pt --prompt "Call me Ishmael." --greedy
```

### 5. Test

```bash
pytest
```

---

## Runtime performance

Training and inference performance settings live in a **separate runtime layer** (`src/transformer/runtime.py`). Model code, loss functions, and the training loop never hardcode hardware assumptions — they receive a `RuntimeConfig` built at startup.

```mermaid
flowchart LR
    Probe[probe_hardware] --> Defaults[default_runtime_config]
    Defaults --> Merge[CLI + JSON overrides]
    Merge --> Apply[apply_runtime_config]
    Apply --> Train[DataLoader + threads + compile]
    Apply --> Infer[KV cache + threads]
```

### What gets tuned automatically (`--runtime-preset auto`)

| Knob | Behavior |
|------|----------|
| **Device** | `cuda` → `mps` → `cpu` (first available) |
| **PyTorch threads** | Matches logical CPU count |
| **DataLoader workers** | ~half of CPU cores (capped at 8), with prefetch |
| **Batch size** | Suggested from available RAM (e.g. 128 on 64+ GB systems) |
| **MKL-DNN** | Enabled on CPU for faster matmuls |
| **KV cache** | Enabled for inference |
| **`torch.compile`** | Off by default; opt in with `--compile` |

### Presets and overrides

| Approach | When to use |
|----------|-------------|
| `--runtime-preset auto` | Default — tune for detected hardware |
| `--runtime-preset minimal` | Portable baseline (no worker processes, no thread overrides) |
| `--runtime-config path.json` | Pin exact settings for your machine or CI |
| `--batch-size N` | Override auto batch size |
| `--num-workers N` | Override DataLoader worker count |
| `--compile` / `--no-compile` | Enable or disable `torch.compile` |

Example `configs/runtime.example.json` (14-core CPU, 128 GB RAM):

```json
{
  "num_threads": 14,
  "num_interop_threads": 4,
  "dataloader_num_workers": 7,
  "dataloader_persistent_workers": true,
  "dataloader_prefetch_factor": 2,
  "use_mkldnn": true,
  "suggested_batch_size": 128,
  "inference_use_kv_cache": true,
  "compile_model": false
}
```

On a GPU machine, a typical override would set `"device": "cuda"`, `"dataloader_pin_memory": true`, and optionally `"compile_model": true`.

### Programmatic use

```python
from transformer.runtime import (
    apply_runtime_config,
    build_runtime_config,
    probe_hardware,
    suggest_batch_size,
)

profile = probe_hardware()
runtime = build_runtime_config(preset="auto", device="auto")
apply_runtime_config(runtime)

batch_size = suggest_batch_size(runtime, model_config)  # or pass explicit override
```

---

## CLI reference

### `scripts/train_book.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | *(required)* | Path to training text file |
| `--out` | `checkpoints` | Directory for saved checkpoints |
| `--resume` | *(none)* | Resume from a checkpoint (e.g. `checkpoints/latest.pt`) |
| `--save-every` | `500` | Save `latest.pt` and a numbered checkpoint every N steps |
| `--epochs` | `3.0` | Full passes over the text |
| `--max-steps` | *(auto)* | Fixed step count; overrides `--epochs` |
| `--batch-size` | auto | Batch size (from hardware profile when omitted) |
| `--block-size` | `256` | Context window (characters) |
| `--d-model` | `128` | Model dimension |
| `--n-heads` | `4` | Attention heads |
| `--n-layers` | `4` | Transformer layers |
| `--lr` | `3e-4` | Learning rate |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |
| `--runtime-preset` | `auto` | `auto` (hardware-tuned) or `minimal` (portable) |
| `--runtime-config` | *(none)* | JSON file overriding `RuntimeConfig` fields |
| `--num-workers` | auto | DataLoader worker processes |
| `--compile` / `--no-compile` | off | Use `torch.compile` when supported |
| `--strip-gutenberg` / `--no-strip-gutenberg` | on | Strip Gutenberg header/footer |

### `scripts/generate.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | *(required)* | Path to `.pt` checkpoint |
| `--prompt` | `""` | Starting text |
| `--max-new-tokens` | `200` | Characters to generate |
| `--temperature` | `0.7` | Sampling temperature |
| `--top-k` | `40` | Limit sampling to top-k logits |
| `--greedy` | off | Greedy decoding (temperature = 0) |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |
| `--runtime-preset` | `auto` | `auto` or `minimal` |
| `--runtime-config` | *(none)* | JSON file overriding `RuntimeConfig` fields |
| `--compile` / `--no-compile` | off | Use `torch.compile` when supported |
| `--no-kv-cache` | off | Disable KV-cache inference (slower; for debugging) |

### Checkpoint contents

Checkpoints are saved **every `--save-every` steps** (default 500), on **normal completion**, and when training is **interrupted** (Ctrl+C or SIGTERM).

```
checkpoints/
├── latest.pt          # always the most recent save (use for generate / resume)
├── step_000500.pt     # numbered snapshot at step 500
├── step_001000.pt     # numbered snapshot at step 1000
└── ...
```

Each `.pt` file stores:

```
├── model_state      # GPT weights
├── model_config     # architecture hyperparameters
├── tokenizer        # char ↔ id mappings
├── optimizer_state  # for resuming training
├── step             # completed training steps
├── max_steps        # target step count
└── eval_loss        # eval loss at save time
```

**Resume after an interrupt:**

```bash
python scripts/train_book.py --data data/book.txt --resume checkpoints/latest.pt
```

To extend training beyond the original target, pass a higher `--max-steps`:

```bash
python scripts/train_book.py --data data/book.txt --resume checkpoints/latest.pt --max-steps 20000
```

---

## Configuration

Configs are split between **model/training hyperparameters** and **runtime performance**:

```python
# src/transformer/config.py — model and training
ModelConfig(vocab_size, d_model, n_heads, n_layers, block_size, dropout)
TrainConfig(lr, batch_size, max_steps, eval_interval, device)

# src/transformer/runtime.py — hardware-aware performance (decoupled)
HardwareProfile(cpu_count, ram_gb, has_cuda, cuda_device_count, has_mps, platform)
RuntimeConfig(device, num_threads, dataloader_num_workers, suggested_batch_size, ...)
```

`RuntimeConfig` fields can be overridden via `--runtime-config` JSON without changing Python code. See [Runtime performance](#runtime-performance).

### Training time estimates (CPU, Moby Dick ~1.2M chars)

Times vary by hardware. With `--runtime-preset auto` on a many-core CPU with ample RAM (larger batch size, parallel data loading), expect noticeably faster steps than the baseline `batch_size=32` setup.

| Setting | Steps | Approx. time (baseline CPU) | Quality |
|---------|-------|----------------------------|---------|
| `--max-steps 1000` | 1,000 | ~10 min | Poor — not enough data seen |
| `--max-steps 5000` | 5,000 | ~1.5 hr | Some structure, still noisy |
| `--epochs 1` | ~38,000 | ~6–7 hr | Decent char-level prose |
| `--epochs 3` (default) | ~115,000 | ~18–20 hr | Best results on CPU |

Char-level models need **many passes** over the text. If output is gibberish, the model is almost certainly under-trained.

---

## Project layout

```
transformer_model/
├── pyproject.toml              # package deps and pytest config
├── README.md
├── license.md                  # MIT License
├── data/
│   └── book.txt                # your training text (not committed)
├── checkpoints/
│   └── latest.pt               # saved after training
├── configs/
│   └── runtime.example.json    # example RuntimeConfig override file
├── src/transformer/
│   ├── config.py               # ModelConfig, TrainConfig
│   ├── runtime.py              # HardwareProfile, RuntimeConfig, auto-tuning
│   ├── tokenizer.py            # CharTokenizer
│   ├── dataset.py              # load, prepare, sliding-window dataset
│   ├── model/
│   │   ├── attention.py        # causal multi-head self-attention (+ KV cache)
│   │   ├── ffn.py              # feed-forward network
│   │   ├── block.py            # pre-norm transformer block
│   │   └── gpt.py              # full GPT model
│   ├── train.py                # loss, train step, checkpoint I/O
│   └── generate.py             # autoregressive sampling (KV-cache path)
├── scripts/
│   ├── train_book.py           # training CLI
│   └── generate.py             # generation CLI
└── tests/
    ├── conftest.py             # shared fixtures
    ├── unit/                   # isolated component tests
    ├── integration/            # wired-together tests
    ├── acceptance/             # end-to-end train + generate
    └── regression/             # fixed-seed golden loss values
```

### Module responsibilities

```mermaid
flowchart LR
    config[config.py] --> gpt[gpt.py]
    config --> train[train.py]
    runtime[runtime.py] --> dataset[dataset.py]
    runtime --> train
    runtime --> generate[generate.py]
    tokenizer[tokenizer.py] --> dataset
    dataset --> train
    attention[attention.py] --> block[block.py]
    ffn[ffn.py] --> block
    block --> gpt
    gpt --> train
    gpt --> generate
    train --> scripts_train[scripts/train_book.py]
    generate --> scripts_gen[scripts/generate.py]
    runtime --> scripts_train
    runtime --> scripts_gen
```

---

## Testing

Tests follow a **four-layer pyramid**:

```mermaid
flowchart TB
    subgraph pyramid [Test pyramid]
        Regression[Regression — fixed-seed golden loss]
        Acceptance[Acceptance — E2E train on tiny corpus]
        Integration[Integration — train step, checkpoint round-trip]
        Unit[Unit — tokenizer, attention, shapes]
    end

    Unit --> Integration --> Acceptance --> Regression
```

| Layer | Directory | What it verifies |
|-------|-----------|------------------|
| **Unit** | `tests/unit/` | Tokenizer, attention mask, shapes, runtime probing, KV-cache parity |
| **Integration** | `tests/integration/` | Gradients flow, checkpoint save/load |
| **Acceptance** | `tests/acceptance/` | Loss decreases on tiny text, generation returns output |
| **Regression** | `tests/regression/` | Exact loss values with `seed=42` to catch silent drift |

Run all tests:

```bash
pytest
pytest -v                    # verbose
pytest tests/unit/           # unit tests only
```

---

## Tips for better output

1. **Train long enough.** For a full book on CPU, aim for at least 1 epoch (`--epochs 1`), ideally 3 (the default).
2. **Let auto-tuning work.** Use `--runtime-preset auto` (the default) so batch size, workers, and threads match your machine. Override only when needed.
3. **Use a book-style prompt.** Match the tone and opening words of your training text (e.g. `"Call me Ishmael."` for Moby Dick).
4. **Try `--greedy` first** when evaluating an under-trained model — sampling adds noise.
5. **Watch eval loss.** It should steadily decrease. If it plateaus above ~2.0, train longer or consider a slightly larger model (`--d-model 256 --n-layers 6`).
6. **Strip boilerplate.** Keep `--strip-gutenberg` on for Project Gutenberg files so the model learns prose, not license text.

---

## Design principles

- **Simplicity first** — char-level tokens, no BPE, no distributed training in v1.
- **Progressive disclosure** — attention → block → GPT → train → generate, one file per concept.
- **Hardware decoupling** — `runtime.py` probes the host and applies performance settings; core model code stays portable.
- **Maintainability** — config dataclasses, small public API, full test pyramid.
- **Readable top-to-bottom** — follow the data from `config.py` through `generate.py`.

### Planned future enhancements

- Subword (BPE) tokenization
- Learning rate warmup / cosine schedule
- Mixed-precision training on supported accelerators

---

## License

This project is licensed under the [MIT License](license.md).

Copyright (c) 2026 Shubham Agarwal.
