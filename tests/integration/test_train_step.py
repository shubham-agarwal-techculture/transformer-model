import torch

from transformer.dataset import create_dataloader
from transformer.model.gpt import GPT
from transformer.train import compute_loss, train_step


def test_train_step_produces_gradients(tiny_corpus, tiny_tokenizer, tiny_config, train_config):
    model = GPT(tiny_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr)
    dataloader = create_dataloader(
        tiny_corpus,
        tiny_tokenizer,
        tiny_config.block_size,
        train_config.batch_size,
        shuffle=False,
    )
    batch = next(iter(dataloader))
    loss = train_step(model, batch, optimizer, train_config.device)

    assert loss > 0
    for param in model.parameters():
        assert param.grad is not None


def test_compute_loss_shape(tiny_config):
    batch, seq_len = 2, 8
    logits = torch.randn(batch, seq_len, tiny_config.vocab_size)
    targets = torch.randint(0, tiny_config.vocab_size, (batch, seq_len))
    loss = compute_loss(logits, targets)
    assert loss.ndim == 0
