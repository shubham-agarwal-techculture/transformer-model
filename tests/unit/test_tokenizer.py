from transformer.tokenizer import CharTokenizer


def test_build_vocab_size(tiny_corpus, tiny_tokenizer):
    assert tiny_tokenizer.vocab_size == len(set(tiny_corpus))


def test_encode_decode_roundtrip(tiny_corpus, tiny_tokenizer):
    ids = tiny_tokenizer.encode(tiny_corpus)
    decoded = tiny_tokenizer.decode(ids)
    assert decoded == tiny_corpus


def test_to_dict_from_dict_roundtrip(tiny_tokenizer):
    restored = CharTokenizer.from_dict(tiny_tokenizer.to_dict())
    text = "hello"
    assert restored.encode(text) == tiny_tokenizer.encode(text)
    assert restored.decode(tiny_tokenizer.encode(text)) == text


def test_unknown_char_uses_fallback():
    tokenizer = CharTokenizer.from_text("abc?")
    ids = tokenizer.encode("a?x")
    assert ids[0] == tokenizer.encode("a")[0]
    assert ids[1] == tokenizer.encode("?")[0]
    assert ids[2] == tokenizer.encode("?")[0]
