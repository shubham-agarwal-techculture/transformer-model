import torch

from transformer.dataset import TextChunkDataset, create_dataloader, load_text


def test_text_chunk_dataset_shapes(tiny_corpus, tiny_tokenizer):
    block_size = 16
    dataset = TextChunkDataset(tiny_corpus, tiny_tokenizer, block_size)
    x, y = dataset[0]
    assert x.shape == (block_size,)
    assert y.shape == (block_size,)
    assert (y[:-1] == x[1:]).all()


def test_dataloader_batch_shape(tiny_corpus, tiny_tokenizer):
    dataloader = create_dataloader(tiny_corpus, tiny_tokenizer, block_size=16, batch_size=4)
    x, y = next(iter(dataloader))
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)


def test_load_text(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("hello world", encoding="utf-8")
    assert load_text(path) == "hello world"
