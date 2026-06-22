import torch

from transformer.model.gpt import GPT
from transformer.train import CheckpointMetadata, load_checkpoint, persist_training_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tiny_config, tiny_tokenizer, tmp_path):
    model = GPT(tiny_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "model.pt"
    metadata = CheckpointMetadata(step=42, max_steps=1000, eval_loss=1.23)

    save_checkpoint(path, model, tiny_config, tiny_tokenizer.to_dict(), optimizer, metadata)
    loaded_model, loaded_config, tokenizer_dict, optimizer_state, loaded_meta = load_checkpoint(path)

    assert loaded_config.vocab_size == tiny_config.vocab_size
    assert loaded_config.d_model == tiny_config.d_model
    assert tokenizer_dict == tiny_tokenizer.to_dict()
    assert optimizer_state is not None
    assert loaded_meta.step == 42
    assert loaded_meta.max_steps == 1000
    assert loaded_meta.eval_loss == 1.23

    idx = torch.randint(0, tiny_config.vocab_size, (1, 4))
    original_logits = model(idx)
    loaded_logits = loaded_model(idx)
    assert torch.allclose(original_logits, loaded_logits)


def test_persist_training_checkpoint_writes_latest_and_numbered(tiny_config, tiny_tokenizer, tmp_path):
    model = GPT(tiny_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    out_dir = tmp_path / "checkpoints"

    latest = persist_training_checkpoint(
        out_dir,
        step=500,
        model=model,
        model_config=tiny_config,
        tokenizer_dict=tiny_tokenizer.to_dict(),
        optimizer=optimizer,
        max_steps=5000,
        eval_loss=2.5,
        numbered=True,
    )

    assert latest == out_dir / "latest.pt"
    assert (out_dir / "latest.pt").exists()
    assert (out_dir / "step_000500.pt").exists()

    _, _, _, _, meta = load_checkpoint(out_dir / "latest.pt")
    assert meta.step == 500
    assert meta.max_steps == 5000
