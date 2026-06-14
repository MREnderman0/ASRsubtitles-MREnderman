from __future__ import annotations

from typing import Any

from rich import print as rprint

from core.utils.config_utils import load_key


_TORCH_CONFIGURED = False


def optional_key(key: str, default: Any = None) -> Any:
    try:
        return load_key(key)
    except KeyError:
        return default


def config_int(key: str, default: int) -> int:
    value = optional_key(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def config_float(key: str, default: float | None = None) -> float | None:
    value = optional_key(key, default)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def config_bool(key: str, default: bool = False) -> bool:
    value = optional_key(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def configure_torch_runtime(torch_module: Any | None = None) -> None:
    global _TORCH_CONFIGURED
    if _TORCH_CONFIGURED:
        return

    if torch_module is None:
        import torch as torch_module

    num_threads = config_int("resource_limits.torch_num_threads", 0)
    if num_threads > 0:
        try:
            torch_module.set_num_threads(num_threads)
            rprint(f"[cyan]Torch CPU threads limited to:[/cyan] {num_threads}")
        except Exception as exc:
            rprint(f"[yellow]Failed to set torch_num_threads: {exc}[/yellow]")

    memory_fraction = config_float("resource_limits.cuda_memory_fraction", None)
    if memory_fraction is not None and 0 < memory_fraction < 1 and torch_module.cuda.is_available():
        try:
            torch_module.cuda.set_per_process_memory_fraction(memory_fraction)
            rprint(f"[cyan]CUDA memory fraction limited to:[/cyan] {memory_fraction:.2f}")
        except Exception as exc:
            rprint(f"[yellow]Failed to set CUDA memory fraction: {exc}[/yellow]")

    _TORCH_CONFIGURED = True


def choose_device(
    section_key: str | None = None,
    *,
    allow_mps: bool = False,
    torch_module: Any | None = None,
) -> str:
    if torch_module is None:
        import torch as torch_module

    section_device = optional_key(section_key, None) if section_key else None
    global_device = optional_key("resource_limits.device", "auto")
    requested = str(section_device or global_device or "auto").strip().lower()
    if requested == "auto":
        if torch_module.cuda.is_available():
            return "cuda"
        if allow_mps and getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda":
        if torch_module.cuda.is_available():
            return "cuda"
        rprint("[yellow]CUDA was requested but is not available; falling back to CPU.[/yellow]")
        return "cpu"

    if requested == "mps":
        if allow_mps and getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
            return "mps"
        rprint("[yellow]MPS was requested but is not available; falling back to CPU.[/yellow]")
        return "cpu"

    if requested == "cpu":
        return "cpu"

    rprint(f"[yellow]Unknown device setting '{requested}'; using auto selection.[/yellow]")
    if torch_module.cuda.is_available():
        return "cuda"
    if allow_mps and getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def whisperx_batch_size(device: str) -> int:
    configured = config_int("resource_limits.whisperx.batch_size", 0)
    if configured > 0:
        return configured
    return 4 if device == "cuda" else 1


def whisperx_compute_type(device: str, torch_module: Any | None = None) -> str:
    configured = str(optional_key("resource_limits.whisperx.compute_type", "int8") or "int8").strip().lower()
    if configured and configured != "auto":
        return configured
    if torch_module is None:
        import torch as torch_module
    return "float16" if device == "cuda" else "int8"
