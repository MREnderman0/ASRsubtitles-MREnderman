from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
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

from core.utils.models import _2_CLEANED_CHUNKS
from subtitle_rag.pipeline import PROJECT_ROOT as PIPELINE_ROOT
from subtitle_rag.pipeline import RUNS_DIR, _load_word_rows, process_words


def main() -> int:
    parser = argparse.ArgumentParser(description="Rerun subtitle processing from an existing word-level ASR xlsx.")
    parser.add_argument(
        "--asr-xlsx",
        default=str(PIPELINE_ROOT / _2_CLEANED_CHUNKS),
        help="Path to cleaned_chunks.xlsx. Defaults to output/log/cleaned_chunks.xlsx.",
    )
    parser.add_argument("--input-path", default="", help="Original media path for manifest only.")
    parser.add_argument("--glossary", action="append", default=[], help="Glossary file path. Can be repeated.")
    parser.add_argument("--reference", action="append", default=[], help="Reference file path. Can be repeated.")
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--overlap-seconds", type=float, default=None)
    parser.add_argument("--max-concurrent-llm-tasks", type=int, default=None)
    parser.add_argument("--disable-global-analysis", action="store_true")
    args = parser.parse_args()

    asr_path = Path(args.asr_xlsx).resolve()
    if not asr_path.exists():
        print(f"ASR xlsx not found: {asr_path}", file=sys.stderr)
        return 2

    run_dir = RUNS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_rerun"
    words = _load_word_rows(asr_path)
    result = process_words(
        words=words,
        run_dir=run_dir,
        input_path=args.input_path or asr_path,
        prepared_media=None,
        glossary_paths=[Path(item) for item in args.glossary],
        reference_paths=[Path(item) for item in args.reference],
        max_chars=args.max_chars,
        window_seconds=args.window_seconds,
        overlap_seconds=args.overlap_seconds,
        max_concurrent_llm_tasks=args.max_concurrent_llm_tasks,
        global_analysis_enabled=not args.disable_global_analysis,
        progress=lambda label, value: print(f"{value:.3f} {label}", flush=True),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
