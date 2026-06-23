import torch
import torch.nn as nn

from transformer.config import ModelConfig
from transformer.model.block import TransformerBlock


class GPT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.block_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model,
                    config.n_heads,
                    config.block_size,
                    config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(
        self,
        idx: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        *,
        use_cache: bool = False,
    ) -> torch.Tensor:
        batch, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block_size {self.config.block_size}")

        if use_cache and kv_caches is not None:
            if seq_len != 1:
                raise ValueError("KV-cache inference expects one new token per forward pass")
            start_pos = kv_caches[0][0].size(2)
            if start_pos >= self.config.block_size:
                raise ValueError(f"Context length {start_pos + 1} exceeds block_size {self.config.block_size}")
            positions = torch.tensor([[start_pos]], device=idx.device, dtype=torch.long)
        else:
            positions = torch.arange(seq_len, device=idx.device).unsqueeze(0)

        x = self.token_emb(idx) + self.pos_emb(positions)
        x = self.drop(x)

        new_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, present_kv = block(x, kv_cache=layer_cache)
            if use_cache:
                new_caches.append(present_kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if use_cache:
            self._last_kv_caches = new_caches

        return logits

    def consume_kv_caches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return KV caches produced by the most recent cached forward pass."""
        caches = getattr(self, "_last_kv_caches", None)
        if caches is None:
            raise RuntimeError("No KV caches available; call forward with kv_caches first")
        return caches
