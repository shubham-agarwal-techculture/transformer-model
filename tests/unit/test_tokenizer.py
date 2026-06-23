from transformer.tokenizer import BPETokenizer, UNK_TOKEN, load_tokenizer


def test_build_vocab_size(tiny_corpus, tiny_tokenizer):
    assert tiny_tokenizer.vocab_size <= 64
    assert tiny_tokenizer.vocab_size >= len(set(tiny_corpus)) + 1


def test_encode_decode_roundtrip(tiny_corpus, tiny_tokenizer):
    ids = tiny_tokenizer.encode(tiny_corpus)
    decoded = tiny_tokenizer.decode(ids)
    assert decoded == tiny_corpus


def test_to_dict_from_dict_roundtrip(tiny_tokenizer):
    restored = BPETokenizer.from_dict(tiny_tokenizer.to_dict())
    text = "hello"
    assert restored.encode(text) == tiny_tokenizer.encode(text)
    assert restored.decode(tiny_tokenizer.encode(text)) == text


def test_deterministic_training(tiny_corpus):
    first = BPETokenizer.from_text(tiny_corpus, vocab_size=64, show_progress=False)
    second = BPETokenizer.from_text(tiny_corpus, vocab_size=64, show_progress=False)
    assert first.to_dict() == second.to_dict()


def test_unknown_char_uses_unk():
    tokenizer = BPETokenizer.from_text("abc", vocab_size=8, show_progress=False)
    ids = tokenizer.encode("a?x")
    unk_id = tokenizer.encode(UNK_TOKEN)[0]
    assert ids[0] == tokenizer.encode("a")[0]
    assert ids[1] == unk_id
    assert ids[2] == unk_id


def test_merges_reduce_sequence_length(tiny_corpus):
    tokenizer = BPETokenizer.from_text(tiny_corpus, vocab_size=64, show_progress=False)
    ids = tokenizer.encode(tiny_corpus)
    assert len(ids) < len(tiny_corpus)


def test_load_tokenizer_rejects_legacy_char_checkpoint():
    import pytest

    with pytest.raises(ValueError, match="legacy character tokenizer"):
        load_tokenizer({"char_to_id": {"a": 0}, "id_to_char": {"0": "a"}})
