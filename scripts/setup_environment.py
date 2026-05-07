from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = PROJECT_ROOT / ".venv"
MODEL_DIR = PROJECT_ROOT / "_model_cache"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
CONFIG_EXAMPLE_PATH = PROJECT_ROOT / "config.example.yaml"

PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
PIP = [str(PYTHON), "-m", "pip"]

WHISPER_REPO_ID = "Huan69/Belle-whisper-large-v3-zh-punct-fasterwhisper"
WHISPER_PROJECT_DIR = MODEL_DIR / "Belle-whisper-large-v3-zh-punct-fasterwhisper"
WHISPER_HF_CACHE_DIR = MODEL_DIR / "models--Huan69--Belle-whisper-large-v3-zh-punct-fasterwhisper"
ALIGN_REPO_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
ALIGN_HF_CACHE_DIR = MODEL_DIR / "models--jonatasgrosman--wav2vec2-large-xlsr-53-chinese-zh-cn"

QWEN_REPO_ID = "Qwen/Qwen3-ASR-1.7B"
QWEN_ALIGNER_REPO_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
QWEN_PROJECT_DIR = MODEL_DIR / "Qwen3-ASR-1.7B"
QWEN_ALIGNER_PROJECT_DIR = MODEL_DIR / "Qwen3-ForcedAligner-0.6B"

REQUIRED_IMPORTS = [
    ("streamlit", "streamlit"),
    ("pandas", "pandas"),
    ("openpyxl", "openpyxl"),
    ("pydub", "pydub"),
    ("openai", "openai"),
    ("json_repair", "json_repair"),
    ("ruamel.yaml", "ruamel"),
    ("whisperx", "whisperx"),
    ("pypdf", "pypdf"),
    ("docx", "python-docx"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize ASR-MREnderman environment without duplicate installs.")
    parser.add_argument("--with-qwen", action="store_true", help="Also check/download Qwen3-ASR model files.")
    parser.add_argument("--skip-models", action="store_true", help="Skip model cache checks and downloads.")
    parser.add_argument("--force-install", action="store_true", help="Run pip install even if imports look available.")
    parser.add_argument("--prefer-mirror", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    ensure_config()
    ensure_venv()
    ensure_pip()
    ensure_dependencies(force=args.force_install)
    check_ffmpeg()
    if not args.skip_models:
        ensure_whisper_models(args.prefer_mirror)
        if args.with_qwen:
            ensure_qwen_models(args.prefer_mirror)
    smoke_test()
    print("\nEnvironment is ready.")
    return 0


def ensure_config() -> None:
    if CONFIG_PATH.exists():
        print("config.yaml exists; skip copy.")
        return
    if not CONFIG_EXAMPLE_PATH.exists():
        raise FileNotFoundError("config.example.yaml is missing.")
    shutil.copy2(CONFIG_EXAMPLE_PATH, CONFIG_PATH)
    print("Created config.yaml from config.example.yaml. Fill API settings before running transcription.")


def ensure_venv() -> None:
    if PYTHON.exists():
        print(f"venv exists: {VENV_DIR}")
        return
    print(f"Creating venv: {VENV_DIR}")
    subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)


def ensure_pip() -> None:
    result = subprocess.run([str(PYTHON), "-m", "pip", "--version"], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout.strip())
        return
    print("pip is missing; installing ensurepip.")
    subprocess.run([str(PYTHON), "-m", "ensurepip", "--upgrade"], check=True)


def ensure_dependencies(force: bool = False) -> None:
    missing = missing_imports()
    if missing:
        print("Missing imports:", ", ".join(missing))
    if not force and not missing:
        print("Python dependencies look installed; skip pip install.")
        return
    print("Installing requirements.txt ...")
    subprocess.run([*PIP, "install", "-r", "requirements.txt"], check=True)


def missing_imports() -> list[str]:
    code = "\n".join(
        [
            "import importlib.util",
            f"items = {REQUIRED_IMPORTS!r}",
            "missing = [pkg for mod, pkg in items if importlib.util.find_spec(mod) is None]",
            "print('\\n'.join(missing))",
        ]
    )
    result = subprocess.run([str(PYTHON), "-c", code], check=True, capture_output=True, text=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg"):
        print("ffmpeg found.")
        return
    print("WARNING: ffmpeg was not found in PATH. Install ffmpeg before processing media.")


def ensure_whisper_models(endpoint: str) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if is_nonempty_dir(WHISPER_PROJECT_DIR) and is_nonempty_dir(ALIGN_HF_CACHE_DIR):
        print("WhisperX model cache exists in project; skip download.")
        return

    found = find_global_hf_cache([WHISPER_REPO_ID, ALIGN_REPO_ID])
    if found and is_nonempty_dir(WHISPER_PROJECT_DIR) and is_nonempty_dir(ALIGN_HF_CACHE_DIR):
        print("WhisperX models found after global cache check; skip download.")
        return

    print("Downloading WhisperX model cache to project _model_cache ...")
    snapshot_download(WHISPER_REPO_ID, MODEL_DIR, endpoint=endpoint)
    snapshot_download(ALIGN_REPO_ID, MODEL_DIR, endpoint=endpoint)


def ensure_qwen_models(endpoint: str) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if is_nonempty_dir(QWEN_PROJECT_DIR) and is_nonempty_dir(QWEN_ALIGNER_PROJECT_DIR):
        print("Qwen3 model directories exist in project; skip download.")
        return

    find_global_hf_cache([QWEN_REPO_ID, QWEN_ALIGNER_REPO_ID])
    if is_nonempty_dir(QWEN_PROJECT_DIR) and is_nonempty_dir(QWEN_ALIGNER_PROJECT_DIR):
        print("Qwen3 models found after global cache check; skip download.")
        return

    print("Downloading Qwen3 model cache to project _model_cache ...")
    snapshot_download(QWEN_REPO_ID, QWEN_PROJECT_DIR, endpoint=endpoint, local_dir=True)
    snapshot_download(QWEN_ALIGNER_REPO_ID, QWEN_ALIGNER_PROJECT_DIR, endpoint=endpoint, local_dir=True)


def is_nonempty_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def find_global_hf_cache(repo_ids: list[str]) -> bool:
    roots = [
        Path(os.environ.get("HF_HOME", "")) / "hub" if os.environ.get("HF_HOME") else None,
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    found_any = False
    for repo_id in repo_ids:
        cache_name = "models--" + repo_id.replace("/", "--")
        for root in roots:
            if root and is_nonempty_dir(root / cache_name):
                print(f"Found global Hugging Face cache: {root / cache_name}")
                found_any = True
                break
    return found_any


def snapshot_download(repo_id: str, target: Path, endpoint: str, local_dir: bool = False) -> None:
    env = os.environ.copy()
    if endpoint:
        env["HF_ENDPOINT"] = endpoint
    env["HF_HUB_DISABLE_XET"] = "1"
    if local_dir:
        code = (
            "from huggingface_hub import snapshot_download; "
            f"snapshot_download({repo_id!r}, local_dir={str(target)!r}, local_dir_use_symlinks=False)"
        )
    else:
        code = (
            "from huggingface_hub import snapshot_download; "
            f"snapshot_download({repo_id!r}, cache_dir={str(target)!r})"
        )
    subprocess.run([str(PYTHON), "-c", code], check=True, env=env)


def smoke_test() -> None:
    code = (
        "from core.utils.config_utils import load_key; "
        "import streamlit, pandas, openpyxl, whisperx, pypdf, docx; "
        "print('smoke_ok', load_key('model_dir'))"
    )
    subprocess.run([str(PYTHON), "-c", code], check=True)


if __name__ == "__main__":
    raise SystemExit(main())
