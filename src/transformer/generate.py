from __future__ import annotations

import torch
import torch.nn.functional as F

from transformer.model.gpt import GPT
from transformer.tokenizer import CharTokenizer


@torch.no_grad()
def generate(
    model: GPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str = "cpu",
    temperature: float = 1.0,
) -> str:
    model.eval()
    ids = tokenizer.encode(prompt)
    block_size = model.config.block_size

    for _ in range(max_new_tokens):
        context = ids[-block_size:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)
        next_logits = logits[0, -1, :]

        if temperature <= 0:
            next_id = int(torch.argmax(next_logits).item())
        else:
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())

        ids.append(next_id)

    return tokenizer.decode(ids)
