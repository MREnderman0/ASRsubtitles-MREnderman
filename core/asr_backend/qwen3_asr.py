from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from rich import print as rprint

from core.utils import except_handler, load_key


_MODEL = None


def _project_path(value: str) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _qwen3_paths() -> tuple[Path, Path]:
    model_dir = _project_path(load_key("model_dir"))
    model_path = _project_path(load_key("asr.qwen3.model_path") or str(model_dir / "Qwen3-ASR-1.7B"))
    aligner_path = _project_path(load_key("asr.qwen3.forced_aligner_path") or str(model_dir / "Qwen3-ForcedAligner-0.6B"))
    return model_path, aligner_path


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    from qwen_asr import Qwen3ASRModel

    model_path, aligner_path = _qwen3_paths()
    if not model_path.exists():
        raise FileNotFoundError(f"Qwen3-ASR model directory not found: {model_path}")
    if not aligner_path.exists():
        raise FileNotFoundError(f"Qwen3 forced aligner directory not found: {aligner_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    batch_size = int(load_key("asr.qwen3.max_inference_batch_size") or 1)
    max_new_tokens = int(load_key("asr.qwen3.max_new_tokens") or 512)
    rprint(f"[cyan]Loading Qwen3-ASR on {device}:[/cyan] {model_path}")
    rprint(f"[cyan]Loading Qwen3 forced aligner:[/cyan] {aligner_path}")

    _MODEL = Qwen3ASRModel.from_pretrained(
        str(model_path),
        forced_aligner=str(aligner_path),
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        max_inference_batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    return _MODEL


@except_handler("Qwen3-ASR processing error:")
def transcribe_audio_qwen3(raw_audio_file: str, vocal_audio_file: str, start: float, end: float) -> dict[str, Any]:
    model = _load_model()
    language = load_key("asr.qwen3.language") or "Chinese"
    context = load_key("asr.qwen3.context") or ""
    audio_file = vocal_audio_file or raw_audio_file
    rprint(f"[green]Starting Qwen3-ASR for segment {start:.2f}s to {end:.2f}s...[/green]")

    result = model.transcribe(
        audio_file,
        context=context,
        language=language,
        return_time_stamps=True,
    )[0]
    words = _items_to_words(result.time_stamps, start=start, end=end)
    text = str(result.text or "").strip()
    if not text:
        text = "".join(word["word"] for word in words)
    if not words and text:
        words = [{"word": text, "start": float(start), "end": float(end)}]
    return {
        "language": "zh",
        "segments": [
            {
                "start": float(start),
                "end": float(end),
                "text": text,
                "words": words,
            }
        ],
    }


def _items_to_words(time_stamps: Any, start: float, end: float) -> list[dict[str, Any]]:
    items = list(getattr(time_stamps, "items", []) or [])
    output: list[dict[str, Any]] = []
    last_end = float(start)
    for item in items:
        text = str(getattr(item, "text", "")).strip()
        if not text:
            continue
        item_start = float(getattr(item, "start_time", last_end))
        item_end = float(getattr(item, "end_time", item_start))
        item_start = max(float(start), item_start + float(start))
        item_end = min(float(end), item_end + float(start))
        if item_end < item_start:
            item_end = item_start
        output.append({"word": text, "start": item_start, "end": item_end})
        last_end = item_end
    return output
