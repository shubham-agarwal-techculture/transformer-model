"""Hardware-aware runtime settings, decoupled from training/inference logic.

Probe the host once, build a :class:`RuntimeConfig`, and pass it into the data
loader, training loop, and generator. Override any field via CLI flags or a
JSON file without touching core model code.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import torch

from transformer.config import ModelConfig


@dataclass(frozen=True)
class HardwareProfile:
    """Read-only facts about the machine. Safe to log or serialize."""

    cpu_count: int
    ram_gb: float | None
    has_cuda: bool
    cuda_device_count: int
    has_mps: bool
    platform: str


@dataclass
class RuntimeConfig:
    """Tunable performance knobs. Defaults are conservative (single-threaded loader)."""

    device: str = "cpu"
    num_threads: int | None = None
    num_interop_threads: int | None = None
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False
    dataloader_persistent_workers: bool = False
    dataloader_prefetch_factor: int | None = None
    use_mkldnn: bool = True
    compile_model: bool = False
    compile_mode: str = "default"
    inference_use_kv_cache: bool = True
    suggested_batch_size: int | None = None


def _read_ram_gb() -> float | None:
    if sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullTotalPhys / (1024**3)
        except (AttributeError, OSError):
            return None
        return None

    if sys.platform == "darwin":
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return (pages * page_size) / (1024**3)
        except (AttributeError, OSError, ValueError):
            return None

    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024**2)
    except OSError:
        return None
    return None


def probe_hardware() -> HardwareProfile:
    """Collect hardware facts without applying any PyTorch settings."""
    return HardwareProfile(
        cpu_count=os.cpu_count() or 1,
        ram_gb=_read_ram_gb(),
        has_cuda=torch.cuda.is_available(),
        cuda_device_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        has_mps=bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        platform=sys.platform,
    )


def _suggest_batch_size(profile: HardwareProfile) -> int:
    if profile.has_cuda:
        return 64
    if profile.ram_gb is not None:
        if profile.ram_gb >= 64:
            return 128
        if profile.ram_gb >= 32:
            return 64
        if profile.ram_gb >= 16:
            return 48
    return 32


def _suggest_num_workers(profile: HardwareProfile) -> int:
    if profile.cpu_count < 4:
        return 0
    # Leave headroom for the main process and BLAS threads.
    return min(8, max(2, profile.cpu_count // 2))


def default_runtime_config(profile: HardwareProfile | None = None) -> RuntimeConfig:
    """Build runtime defaults from a hardware profile (auto-detected when omitted)."""
    profile = profile or probe_hardware()

    device = "cuda" if profile.has_cuda else ("mps" if profile.has_mps else "cpu")
    num_workers = _suggest_num_workers(profile)
    pin_memory = profile.has_cuda

    return RuntimeConfig(
        device=device,
        num_threads=profile.cpu_count,
        num_interop_threads=min(4, max(2, profile.cpu_count // 4)),
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=pin_memory,
        dataloader_persistent_workers=num_workers > 0,
        dataloader_prefetch_factor=2 if num_workers > 0 else None,
        use_mkldnn=True,
        compile_model=False,
        inference_use_kv_cache=True,
        suggested_batch_size=_suggest_batch_size(profile),
    )


def minimal_runtime_config(device: str = "cpu") -> RuntimeConfig:
    """Portable baseline: no thread overrides, no background data loading."""
    return RuntimeConfig(device=device)


def resolve_device(requested: str | None, profile: HardwareProfile | None = None) -> str:
    """Map ``auto`` / ``None`` to the best available device."""
    if requested and requested not in ("auto", ""):
        return requested
    profile = profile or probe_hardware()
    if profile.has_cuda:
        return "cuda"
    if profile.has_mps:
        return "mps"
    return "cpu"


def apply_runtime_config(config: RuntimeConfig) -> None:
    """Apply process-wide PyTorch settings. Call once before training or inference."""
    if config.num_threads is not None:
        torch.set_num_threads(config.num_threads)
    if config.num_interop_threads is not None:
        torch.set_num_interop_threads(config.num_interop_threads)
    if config.use_mkldnn and hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = True


def maybe_compile_model(model: torch.nn.Module, config: RuntimeConfig) -> torch.nn.Module:
    if not config.compile_model or not hasattr(torch, "compile"):
        return model
    return torch.compile(model, mode=config.compile_mode)


def dataloader_kwargs(config: RuntimeConfig) -> dict[str, Any]:
    """Keyword arguments for :class:`torch.utils.data.DataLoader`."""
    kwargs: dict[str, Any] = {
        "num_workers": config.dataloader_num_workers,
        "pin_memory": config.dataloader_pin_memory,
    }
    if config.dataloader_num_workers > 0:
        kwargs["persistent_workers"] = config.dataloader_persistent_workers
        if config.dataloader_prefetch_factor is not None:
            kwargs["prefetch_factor"] = config.dataloader_prefetch_factor
    return kwargs


def batch_transfer_kwargs(config: RuntimeConfig) -> dict[str, Any]:
    """``Tensor.to`` kwargs for host → device copies during training."""
    use_non_blocking = config.dataloader_pin_memory and config.device.startswith("cuda")
    return {"non_blocking": use_non_blocking}


def runtime_config_from_dict(data: dict[str, Any]) -> RuntimeConfig:
    valid = {f.name for f in fields(RuntimeConfig)}
    return RuntimeConfig(**{k: v for k, v in data.items() if k in valid})


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Runtime config JSON must be an object")
    return runtime_config_from_dict(payload)


def merge_runtime_config(base: RuntimeConfig, overrides: dict[str, Any]) -> RuntimeConfig:
    merged = asdict(base)
    valid = {f.name for f in fields(RuntimeConfig)}
    merged.update({k: v for k, v in overrides.items() if k in valid})
    return RuntimeConfig(**merged)


def build_runtime_config(
    *,
    profile: HardwareProfile | None = None,
    preset: str = "auto",
    device: str | None = None,
    config_path: str | Path | None = None,
    compile_model: bool | None = None,
    num_workers: int | None = None,
) -> RuntimeConfig:
    """Compose a runtime config from preset, hardware, file, and CLI overrides."""
    profile = profile or probe_hardware()

    if preset == "minimal":
        base = minimal_runtime_config(device=resolve_device(device, profile))
    else:
        base = default_runtime_config(profile)

    if device is not None:
        base = replace(base, device=resolve_device(device, profile))

    if compile_model is not None:
        base = replace(base, compile_model=compile_model)
    if num_workers is not None:
        base = replace(
            base,
            dataloader_num_workers=num_workers,
            dataloader_persistent_workers=num_workers > 0,
            dataloader_prefetch_factor=2 if num_workers > 0 else None,
        )

    if config_path is not None:
        file_config = load_runtime_config(config_path)
        base = merge_runtime_config(base, asdict(file_config))

    return base


def format_runtime_summary(config: RuntimeConfig, profile: HardwareProfile) -> str:
    ram = f"{profile.ram_gb:.0f} GB" if profile.ram_gb is not None else "unknown"
    lines = [
        f"Platform: {profile.platform}, CPUs: {profile.cpu_count}, RAM: {ram}",
        f"Accelerators: cuda={profile.has_cuda} (n={profile.cuda_device_count}), mps={profile.has_mps}",
        f"Device: {config.device}, threads={config.num_threads}, interop={config.num_interop_threads}",
        (
            f"DataLoader: workers={config.dataloader_num_workers}, "
            f"pin_memory={config.dataloader_pin_memory}, prefetch={config.dataloader_prefetch_factor}"
        ),
        f"Suggested batch size: {config.suggested_batch_size}, compile={config.compile_model}",
        f"Inference KV cache: {config.inference_use_kv_cache}",
    ]
    return "\n".join(lines)


def suggest_batch_size(
    runtime: RuntimeConfig,
    model_config: ModelConfig,
    requested: int | None = None,
) -> int:
    """Pick batch size: explicit CLI value wins, else runtime suggestion."""
    if requested is not None:
        return requested
    if runtime.suggested_batch_size is not None:
        return runtime.suggested_batch_size
    return 32
