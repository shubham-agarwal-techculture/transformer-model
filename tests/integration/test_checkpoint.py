import torch

from transformer.model.gpt import GPT
from transformer.train import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tiny_config, tiny_tokenizer, tmp_path):
    model = GPT(tiny_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "model.pt"

    save_checkpoint(path, model, tiny_config, tiny_tokenizer.to_dict(), optimizer)
    loaded_model, loaded_config, tokenizer_dict, optimizer_state = load_checkpoint(path)

    assert loaded_config.vocab_size == tiny_config.vocab_size
    assert loaded_config.d_model == tiny_config.d_model
    assert tokenizer_dict == tiny_tokenizer.to_dict()
    assert optimizer_state is not None

    idx = torch.randint(0, tiny_config.vocab_size, (1, 4))
    original_logits = model(idx)
    loaded_logits = loaded_model(idx)
    assert torch.allclose(original_logits, loaded_logits)
