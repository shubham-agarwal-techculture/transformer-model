import torch

from transformer.model.attention import CausalSelfAttention


def test_attention_output_shape():
    batch, seq_len, d_model, n_heads = 2, 8, 32, 4
    attn = CausalSelfAttention(d_model, n_heads, block_size=16, dropout=0.0)
    x = torch.randn(batch, seq_len, d_model)
    out = attn(x)
    assert out.shape == (batch, seq_len, d_model)


def test_causal_mask_is_lower_triangular():
    attn = CausalSelfAttention(d_model=32, n_heads=4, block_size=8, dropout=0.0)
    mask = attn.causal_mask[0, 0]
    for i in range(8):
        for j in range(8):
            if j > i:
                assert mask[i, j] == 0
            else:
                assert mask[i, j] == 1


def test_future_tokens_do_not_affect_past_outputs():
    attn = CausalSelfAttention(d_model=32, n_heads=4, block_size=8, dropout=0.0)
    attn.eval()
    x1 = torch.randn(1, 4, 32)
    x2 = x1.clone()
    x2[0, 3, :] = torch.randn(32) * 100

    with torch.no_grad():
        out1 = attn(x1)
        out2 = attn(x2)

    assert torch.allclose(out1[0, :3, :], out2[0, :3, :])
    assert not torch.allclose(out1[0, 3, :], out2[0, 3, :])
