from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from core import _2_asr
from core.utils.config_utils import load_key
from core.utils.models import _2_CLEANED_CHUNKS
from subtitle_rag.cleaning import CleanedSegment, Segment, deterministic_clean
from subtitle_rag.patching import review_and_apply_patches
from subtitle_rag.planning import plan_segments_from_words
from subtitle_rag.parsers import read_glossary_files, read_reference_files
from subtitle_rag.rag import RagContext, apply_glossary
from subtitle_rag.subtitle import display_len, make_subtitles, write_srt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTENSION_ROOT = Path(__file__).resolve().parent
RUNS_DIR = EXTENSION_ROOT / "runs"
OUTPUT_DIR = PROJECT_ROOT / "output"
os.chdir(PROJECT_ROOT)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("RICH_NO_LEGACY_WINDOWS", "1")
os.environ.setdefault("TERM", "xterm-256color")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ProgressCallback = Callable[[str, float], None]


class SubtitleRagError(RuntimeError):
    """User-facing error raised by the subtitle RAG pipeline."""


def process_media(
    input_path: str | Path,
    glossary_paths: list[str | Path] | None = None,
    reference_paths: list[str | Path] | None = None,
    max_chars: int | None = None,
    window_seconds: float | None = None,
    overlap_seconds: float | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, str]:
    glossary_paths = glossary_paths or []
    reference_paths = reference_paths or []
    max_chars = max(int(max_chars if max_chars is not None else _config_default("subtitle_rag.max_chars", 17)), 1)
    window_seconds = max(float(window_seconds if window_seconds is not None else _config_default("subtitle_rag.window_seconds", 600)), 1.0)
    overlap_seconds = max(float(overlap_seconds if overlap_seconds is not None else _config_default("subtitle_rag.overlap_seconds", 30)), 0.0)

    run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    _progress(progress, "Preparing input", 0.05)
    prepared_media = _prepare_input(Path(input_path), run_dir)

    _progress(progress, "Running local ASR word-level transcription", 0.18)
    _run_asr()

    _progress(progress, "Loading word-level timestamps", 0.48)
    words = _load_word_rows(PROJECT_ROOT / _2_CLEANED_CHUNKS)

    _progress(progress, "Planning subtitle boundaries with LLM", 0.53)
    segments, boundary_plan_stats = _plan_segments(
        words,
        max_chars=max_chars,
        run_dir=run_dir,
        window_seconds=window_seconds,
        overlap_seconds=overlap_seconds,
        progress=progress,
    )
    if boundary_plan_stats.get("boundary_plan_fallback_used"):
        failed = boundary_plan_stats.get("boundary_plan_failed_count", 0)
        error = boundary_plan_stats.get("boundary_plan_error", "")
        detail = f"{failed} block(s) failed" if failed else str(error or "fallback used")
        _progress(progress, f"FAIL:LLM 规划字幕边界失败，已回退本地断句（{detail}）", 0.535)

    _progress(progress, "Loading glossary and reference materials", 0.58)
    glossary = read_glossary_files(glossary_paths)
    references = read_reference_files(reference_paths)
    rag_context = RagContext(glossary=glossary, references=references)

    _progress(progress, "Generating draft transcript", 0.70)
    cleaned = _draft_clean_segments(segments, rag_context)

    _progress(progress, "Generating draft SRT subtitles", 0.82)
    draft_subtitles = make_subtitles(cleaned, max_chars=max_chars, protected_phrases=_protected_phrases_from_context(rag_context))

    _progress(progress, "Reviewing draft with LLM patches", 0.88)
    subtitles, patch_uncertain, patch_stats = review_and_apply_patches(
        draft_items=draft_subtitles,
        segments=cleaned,
        rag_context=rag_context,
        max_chars=max_chars,
        window_seconds=window_seconds,
        overlap_seconds=overlap_seconds,
        run_dir=run_dir,
        progress=progress,
    )

    final_srt = run_dir / "final.srt"
    draft_srt = run_dir / "draft.srt"
    cleaned_xlsx = run_dir / "cleaned_subtitles.xlsx"
    uncertain_csv = run_dir / "uncertain_terms.csv"
    manifest_path = run_dir / "run_manifest.json"

    write_srt(draft_subtitles, draft_srt)
    write_srt(subtitles, final_srt)
    _write_cleaned_xlsx(cleaned, subtitles, cleaned_xlsx)
    _write_uncertain(cleaned, uncertain_csv, patch_uncertain)
    _write_manifest(
        manifest_path,
        input_path=Path(input_path),
        prepared_media=prepared_media,
        max_chars=max_chars,
        word_count=len(words),
        segment_count=len(cleaned),
        draft_subtitle_count=len(draft_subtitles),
        subtitle_count=len(subtitles),
        glossary_count=len(glossary),
        reference_count=len(references),
        draft_srt=draft_srt,
        **boundary_plan_stats,
        **patch_stats,
    )
    _archive_asr_outputs(run_dir)

    zip_path = _write_result_zip(run_dir)
    _progress(progress, "Done", 1.0)

    return {
        "run_dir": str(run_dir),
        "draft_srt": str(draft_srt),
        "final_srt": str(final_srt),
        "cleaned_subtitles": str(cleaned_xlsx),
        "uncertain_terms": str(uncertain_csv),
        "manifest": str(manifest_path),
        "zip": zip_path,
    }


def _config_default(key: str, fallback):
    try:
        return load_key(key)
    except Exception:
        return fallback


def _plan_segments(
    words: pd.DataFrame,
    max_chars: int,
    run_dir: Path,
    window_seconds: float,
    overlap_seconds: float,
    progress: ProgressCallback | None,
) -> tuple[list[Segment], dict]:
    try:
        segments, stats = plan_segments_from_words(
            words,
            max_chars=max_chars,
            run_dir=run_dir,
            window_seconds=window_seconds,
            overlap_seconds=overlap_seconds,
            progress=progress,
        )
        if segments:
            return segments, stats
    except Exception as exc:
        stats = {
            "boundary_plan_enabled": True,
            "boundary_plan_fallback_used": True,
            "boundary_plan_error": str(exc),
        }
    segments = _segments_from_words(words)
    stats.setdefault("boundary_plan_segment_count", len(segments))
    return segments, stats


def _draft_clean_segments(segments: list[Segment], rag_context: RagContext) -> list[CleanedSegment]:
    cleaned: list[CleanedSegment] = []
    for seg in segments:
        corrected, _ = apply_glossary(seg.text, rag_context.glossary)
        corrected = _apply_glossary_fuzzy(corrected, rag_context)
        cleaned.append(
            CleanedSegment(
                id=seg.id,
                start=seg.start,
                end=seg.end,
                raw_text=seg.text,
                cleaned_text=deterministic_clean(corrected),
                tokens=seg.tokens,
            )
        )
    return cleaned


def _apply_glossary_fuzzy(text: str, rag_context: RagContext) -> str:
    corrected = str(text or "")
    for entry in sorted(rag_context.glossary, key=lambda item: len(item.canonical), reverse=True):
        canonical = str(entry.canonical or "").strip()
        if len(canonical) < 3 or canonical in corrected:
            continue
        for candidate in _same_length_cjk_windows(corrected, len(canonical)):
            if _one_char_different(candidate, canonical):
                corrected = corrected.replace(candidate, canonical)
    return corrected


def _same_length_cjk_windows(text: str, size: int) -> list[str]:
    windows: list[str] = []
    for start in range(0, max(len(text) - size + 1, 0)):
        candidate = text[start : start + size]
        if all("\u4e00" <= char <= "\u9fff" for char in candidate):
            windows.append(candidate)
    return windows


def _one_char_different(left: str, right: str) -> bool:
    if len(left) != len(right):
        return False
    return sum(1 for a, b in zip(left, right) if a != b) == 1


def _prepare_input(input_path: Path, run_dir: Path) -> Path:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = _safe_filename(input_path.name)
    copied = OUTPUT_DIR / safe_name
    shutil.copy2(input_path, copied)
    (run_dir / "input").mkdir(exist_ok=True)
    shutil.copy2(input_path, run_dir / "input" / safe_name)

    suffix = copied.suffix.lower().lstrip(".")
    if suffix in set(load_key("allowed_audio_formats")):
        video_path = OUTPUT_DIR / "black_screen.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=1280x720:r=25",
            "-i",
            str(copied),
            "-shortest",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        copied.unlink(missing_ok=True)
        return video_path
    return copied


def _run_asr() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("RICH_NO_LEGACY_WINDOWS", "1")
    os.environ.setdefault("TERM", "xterm-256color")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["VIDEOLINGO_KEEP_HF_ENDPOINT"] = "1"
    os.environ["VIDEOLINGO_ASR_LOCAL_ONLY"] = "1"
    _ensure_nltk_punkt_tab()
    try:
        _2_asr.transcribe()
    except Exception as exc:
        message = str(exc)
        error_type = exc.__class__.__name__
        download_markers = (
            "LocalEntryNotFoundError",
            "ConnectionResetError",
            "Connection aborted",
            "snapshot_download",
            "huggingface",
            "Hub",
        )
        if error_type == "LocalEntryNotFoundError" or any(marker in message for marker in download_markers):
            raise SubtitleRagError(
                "WhisperX 模型下载失败。本地没有 `_model_cache` 缓存，当前网络连接 HuggingFace/hf-mirror 时中断。"
                "请切换网络后重试，或先在 VideoLingo 主项目中完成一次 WhisperX 模型下载。"
            ) from exc
        raise


def _ensure_nltk_punkt_tab() -> None:
    try:
        import nltk.data

        nltk.data.find("tokenizers/punkt_tab/english/")
        return
    except LookupError:
        pass

    try:
        import pickle
        from pathlib import Path
        from nltk.tokenize.punkt import save_punkt_params
        import nltk.data

        punkt_path = Path(nltk.data.find("tokenizers/punkt/english.pickle"))
        nltk_root = punkt_path.parents[1]
        out_dir = nltk_root / "punkt_tab" / "english"
        out_dir.mkdir(parents=True, exist_ok=True)
        with punkt_path.open("rb") as file:
            tokenizer = pickle.load(file)
        save_punkt_params(tokenizer._params, dir=str(out_dir))
        nltk.data.find("tokenizers/punkt_tab/english/")
    except Exception as exc:
        raise SubtitleRagError(
            "缺少 NLTK punkt_tab 分句资源，且无法从本地 punkt 数据自动生成。"
            "请在虚拟环境中运行：python -m nltk.downloader punkt punkt_tab"
        ) from exc


def _load_word_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"ASR output not found: {path}")
    frame = pd.read_excel(path)
    required = {"text", "start", "end"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ASR output is missing columns: {', '.join(sorted(missing))}")
    frame = frame.copy()
    frame["text"] = frame["text"].astype(str).str.strip().str.strip('"').str.strip()
    frame = frame[frame["text"].str.len() > 0].reset_index(drop=True)
    return frame


def _segments_from_words(words: pd.DataFrame, max_words: int = 42) -> list[Segment]:
    segments: list[Segment] = []
    buffer: list[dict] = []
    sentence_marks = tuple("。！？!?；;")
    comma_marks = tuple("，,、")

    for _, row in words.iterrows():
        item = {"text": str(row["text"]), "start": float(row["start"]), "end": float(row["end"])}
        buffer.append(item)
        should_break = item["text"].endswith(sentence_marks)
        if len(buffer) >= max_words and item["text"].endswith(comma_marks):
            should_break = True
        if len(buffer) >= max_words + 18:
            should_break = True
        if should_break:
            _append_segment(segments, buffer)
            buffer = []
    if buffer:
        _append_segment(segments, buffer)
    return segments


def _append_segment(segments: list[Segment], buffer: list[dict]) -> None:
    text = _join_words([item["text"] for item in buffer])
    if text:
        segments.append(
            Segment(
                id=len(segments) + 1,
                start=buffer[0]["start"],
                end=buffer[-1]["end"],
                text=text,
                tokens=[dict(item) for item in buffer],
            )
        )


def _join_words(words: list[str]) -> str:
    text = ""
    for word in words:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-']*", word):
            text = f"{text} {word}".strip()
        else:
            text += word
    return re.sub(r"\s+([，。！？；：,.!?;:])", r"\1", text).strip()


def _write_cleaned_xlsx(cleaned: list[CleanedSegment], subtitles, path: Path) -> None:
    segment_rows = [
        {
            "id": item.id,
            "start": item.start,
            "end": item.end,
            "raw_text": item.raw_text,
            "cleaned_text": item.cleaned_text,
        }
        for item in cleaned
    ]
    subtitle_rows = [
        {
            "index": item.index,
            "start": item.start,
            "end": item.end,
            "text": item.text,
            "display_len": display_len(item.text),
        }
        for item in subtitles
    ]
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(segment_rows).to_excel(writer, sheet_name="cleaned_segments", index=False)
        pd.DataFrame(subtitle_rows).to_excel(writer, sheet_name="subtitles", index=False)


def _write_uncertain(cleaned: list[CleanedSegment], path: Path, extra_rows: list[dict] | None = None) -> None:
    rows = []
    for item in cleaned:
        for uncertain in item.uncertain_terms:
            row = {
                "start_time": uncertain.get("start_time", item.start),
                "end_time": uncertain.get("end_time", item.end),
                "raw_asr_text": uncertain.get("raw_asr_text", item.raw_text),
                "suggested_text": uncertain.get("suggested_text", ""),
                "reason": uncertain.get("reason", ""),
                "source": uncertain.get("source", "unresolved"),
                "confidence": uncertain.get("confidence", ""),
            }
            rows.append(row)
    rows.extend(extra_rows or [])
    pd.DataFrame(
        rows,
        columns=["start_time", "end_time", "raw_asr_text", "suggested_text", "reason", "source", "confidence"],
    ).to_csv(path, index=False, encoding="utf-8-sig")


def _protected_phrases_from_context(rag_context: RagContext) -> list[str]:
    phrases: set[str] = set()
    for entry in rag_context.glossary:
        for value in (entry.alias, entry.canonical):
            phrase = str(value or "").strip()
            if 2 <= len(phrase) <= 30:
                phrases.add(phrase)
    return sorted(phrases, key=len, reverse=True)


def _write_manifest(path: Path, **values) -> None:
    serializable = {key: str(value) if isinstance(value, Path) else value for key, value in values.items()}
    serializable["created_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def _archive_asr_outputs(run_dir: Path) -> None:
    log_dir = OUTPUT_DIR / "log"
    if log_dir.exists():
        shutil.copytree(log_dir, run_dir / "asr_log", dirs_exist_ok=True)


def _write_result_zip(run_dir: Path) -> str:
    zip_path = run_dir / "subtitle_rag_result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in run_dir.rglob("*"):
            if path == zip_path or not path.is_file():
                continue
            archive.write(path, path.relative_to(run_dir))
    return str(zip_path)


def _safe_filename(name: str) -> str:
    path = Path(name)
    stem = re.sub(r"[^\w.\-]+", "_", path.stem, flags=re.UNICODE).strip("._") or "input"
    suffix = re.sub(r"[^\w.]+", "", path.suffix.lower())
    return f"{stem}{suffix}"


def _progress(callback: ProgressCallback | None, label: str, value: float) -> None:
    if callback:
        callback(label, value)
