import json

import torch

from transformer.config import ModelConfig
from transformer.generate import generate
from transformer.model.gpt import GPT
from transformer.runtime import (
    RuntimeConfig,
    build_runtime_config,
    default_runtime_config,
    load_runtime_config,
    merge_runtime_config,
    probe_hardware,
    suggest_batch_size,
)


def test_probe_hardware_returns_positive_cpu_count():
    profile = probe_hardware()
    assert profile.cpu_count >= 1
    assert profile.platform


def test_default_runtime_config_for_high_ram_cpu():
    profile = probe_hardware()
    profile = profile.__class__(
        cpu_count=14,
        ram_gb=128.0,
        has_cuda=False,
        cuda_device_count=0,
        has_mps=False,
        platform="win32",
    )
    runtime = default_runtime_config(profile)
    assert runtime.device == "cpu"
    assert runtime.num_threads == 14
    assert runtime.dataloader_num_workers >= 2
    assert runtime.suggested_batch_size == 128


def test_build_runtime_config_minimal_preset():
    runtime = build_runtime_config(preset="minimal", device="cpu")
    assert runtime.device == "cpu"
    assert runtime.dataloader_num_workers == 0
    assert runtime.num_threads is None


def test_runtime_config_json_round_trip(tmp_path):
    path = tmp_path / "runtime.json"
    path.write_text(
        json.dumps({"dataloader_num_workers": 4, "compile_model": True}),
        encoding="utf-8",
    )
    loaded = load_runtime_config(path)
    assert loaded.dataloader_num_workers == 4
    assert loaded.compile_model is True

    merged = merge_runtime_config(default_runtime_config(), {"suggested_batch_size": 96})
    assert merged.suggested_batch_size == 96


def test_suggest_batch_size_prefers_explicit_request():
    runtime = RuntimeConfig(suggested_batch_size=128)
    model_config = ModelConfig()
    assert suggest_batch_size(runtime, model_config, requested=16) == 16
    assert suggest_batch_size(runtime, model_config) == 128


def test_kv_cache_greedy_matches_naive(tiny_corpus, tiny_tokenizer, tiny_config):
    torch.manual_seed(0)
    model = GPT(tiny_config)
    model.eval()
    prompt = tiny_corpus[:24]

    greedy = generate(
        model,
        tiny_tokenizer,
        prompt,
        max_new_tokens=12,
        device="cpu",
        temperature=0.0,
        top_k=None,
        use_kv_cache=False,
    )
    cached = generate(
        model,
        tiny_tokenizer,
        prompt,
        max_new_tokens=12,
        device="cpu",
        temperature=0.0,
        top_k=None,
        use_kv_cache=True,
    )
    assert greedy == cached
