from pathlib import Path

import torch

from transformer.config import ModelConfig, TrainConfig
from transformer.dataset import create_dataloader, load_text
from transformer.generate import generate
from transformer.model.gpt import GPT
from transformer.tokenizer import BPETokenizer
from transformer.train import evaluate_loss, train_step


FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "tiny_book.txt"


def test_train_and_generate_on_tiny_corpus():
    text = load_text(FIXTURE_PATH)
    tokenizer = BPETokenizer.from_text(text, vocab_size=64, show_progress=False)
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_heads=4,
        n_layers=2,
        block_size=32,
        dropout=0.0,
    )
    train_config = TrainConfig(lr=3e-3, batch_size=4, max_steps=30, device="cpu")

    dataloader = create_dataloader(
        text, tokenizer, model_config.block_size, train_config.batch_size
    )
    model = GPT(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr)

    initial_loss = evaluate_loss(model, dataloader, train_config.device, max_batches=3)

    for batch in dataloader:
        train_step(model, batch, optimizer, train_config.device)

    final_loss = evaluate_loss(model, dataloader, train_config.device, max_batches=3)
    assert final_loss < initial_loss

    output = generate(
        model,
        tokenizer,
        prompt="The",
        max_new_tokens=20,
        device="cpu",
        temperature=0.0,
    )
    assert len(output) > len("The")
