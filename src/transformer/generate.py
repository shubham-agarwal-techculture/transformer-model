from __future__ import annotations

import torch
import torch.nn.functional as F

from transformer.model.gpt import GPT
from transformer.tokenizer import CharTokenizer


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    scaled = logits / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, scaled.size(-1))
        cutoff = torch.topk(scaled, k).values[-1]
        scaled = scaled.masked_fill(scaled < cutoff, float("-inf"))

    probs = F.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate(
    model: GPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str = "cpu",
    temperature: float = 1.0,
    top_k: int | None = 40,
) -> str:
    model.eval()
    ids = tokenizer.encode(prompt)
    block_size = model.config.block_size

    for _ in range(max_new_tokens):
        context = ids[-block_size:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)
        next_id = sample_next_token(logits[0, -1, :], temperature=temperature, top_k=top_k)
        ids.append(next_id)

    return tokenizer.decode(ids)
