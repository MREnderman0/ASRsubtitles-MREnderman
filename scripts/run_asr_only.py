from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("RICH_NO_LEGACY_WINDOWS", "1")
os.environ.setdefault("TERM", "xterm-256color")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["VIDEOLINGO_KEEP_HF_ENDPOINT"] = "1"
os.environ["VIDEOLINGO_ASR_LOCAL_ONLY"] = "1"

from core import _2_asr
from core.utils.models import _2_CLEANED_CHUNKS, _2_QWEN_RAW_RESULTS
from subtitle_rag.pipeline import OUTPUT_DIR, _safe_filename


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/run_asr_only.py <audio-or-video-path> [run-dir]", file=sys.stderr)
        return 2

    input_path = Path(sys.argv[1]).resolve()
    if not input_path.exists():
        print(f"input not found: {input_path}", file=sys.stderr)
        return 2

    run_dir = Path(sys.argv[2]).resolve() if len(sys.argv) >= 3 else None
    _prepare_output(input_path)
    _2_asr.transcribe()

    cleaned_path = PROJECT_ROOT / _2_CLEANED_CHUNKS
    raw_path = PROJECT_ROOT / _2_QWEN_RAW_RESULTS
    summary = _summarize_asr(cleaned_path, raw_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if run_dir:
        target = run_dir / "asr_only_check"
        target.mkdir(parents=True, exist_ok=True)
        if cleaned_path.exists():
            shutil.copy2(cleaned_path, target / cleaned_path.name)
        if raw_path.exists():
            shutil.copy2(raw_path, target / raw_path.name)
        (target / "asr_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {target}")
    return 0


def _prepare_output(input_path: Path) -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, OUTPUT_DIR / _safe_filename(input_path.name))


def _summarize_asr(cleaned_path: Path, raw_path: Path) -> dict:
    summary: dict = {
        "cleaned_chunks": str(cleaned_path),
        "qwen_raw_results": str(raw_path),
        "cleaned_exists": cleaned_path.exists(),
        "qwen_raw_exists": raw_path.exists(),
        "row_count": 0,
        "large_gaps": [],
        "raw_segment_count": 0,
        "raw_timestamp_count": 0,
    }
    if cleaned_path.exists():
        df = pd.read_excel(cleaned_path)
        summary["row_count"] = int(len(df))
        gaps = []
        previous = None
        for idx, row in df.iterrows():
            if previous is not None:
                gap = float(row["start"]) - float(previous["end"])
                if gap > 3:
                    gaps.append(
                        {
                            "row": int(idx),
                            "gap_seconds": round(gap, 3),
                            "previous_text": str(previous["text"]),
                            "previous_end": float(previous["end"]),
                            "next_text": str(row["text"]),
                            "next_start": float(row["start"]),
                        }
                    )
            previous = row
        summary["large_gaps"] = gaps
    if raw_path.exists():
        raw_segment_count = 0
        raw_timestamp_count = 0
        with raw_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                raw_segment_count += 1
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw_timestamp_count += len(payload.get("time_stamps", []) or [])
        summary["raw_segment_count"] = raw_segment_count
        summary["raw_timestamp_count"] = raw_timestamp_count
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
