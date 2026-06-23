import torch

from transformer.dataset import create_dataloader
from transformer.model.gpt import GPT
from transformer.train import compute_loss, train_step

# Golden values captured with torch.manual_seed(42) and the tiny fixtures in conftest.
GOLDEN_INITIAL_LOSS = 22.701580047607422
GOLDEN_POST_STEP_LOSS = 21.963275909423828


def test_fixed_seed_initial_loss(tiny_corpus, tiny_tokenizer, tiny_config, train_config):
    torch.manual_seed(42)
    model = GPT(tiny_config)
    dataloader = create_dataloader(
        tiny_corpus,
        tiny_tokenizer,
        tiny_config.block_size,
        train_config.batch_size,
        shuffle=False,
    )
    batch = next(iter(dataloader))
    logits = model(batch[0])
    loss = compute_loss(logits, batch[1])

    assert logits.shape == (
        train_config.batch_size,
        tiny_config.block_size,
        tiny_config.vocab_size,
    )
    assert abs(loss.item() - GOLDEN_INITIAL_LOSS) < 1e-5


def test_fixed_seed_post_step_loss(tiny_corpus, tiny_tokenizer, tiny_config, train_config):
    torch.manual_seed(42)
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
    train_step(model, batch, optimizer, train_config.device)

    logits = model(batch[0])
    loss = compute_loss(logits, batch[1])
    assert abs(loss.item() - GOLDEN_POST_STEP_LOSS) < 1e-5
