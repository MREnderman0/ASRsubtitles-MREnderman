# ASR-MREnderman

ASR-MREnderman is a local subtitle generation workflow for audio and video files. It combines local ASR, word-level timestamps, LLM-assisted subtitle boundary planning, and LLM content review to produce clean source-language SRT subtitles.

## 功能

- 上传本地音频或视频并生成原文字幕。
- 使用本地 ASR 生成词级或字级时间戳。
- 支持 WhisperX 本地识别，保留 Qwen3-ASR 作为可选后端。
- 可使用 Demucs 做人声分离，提升复杂音频中的识别效果。
- 通过 LLM 规划字幕短语边界，减少断词和不自然拆分。
- 在内容校对前对完整 ASR 转录做一次全局诊断，生成主题、术语和疑似识别错误资料。
- 生成 `draft.srt` 后再由 LLM 做内容校对。
- 支持自定义词汇表，优先修正专业术语、人名、机构名和固定表达。
- 支持参考资料上传，内容校对阶段会把参考资料完整提供给 LLM。
- 支持下载 `final.srt`、`draft.srt`、待确认词表、patch 报告和完整结果包。
- 支持前端页面使用，也支持 agent 或脚本直接调用 Python API。

## 安装配置环境

### 1. 准备 Python

建议使用 Python 3.10 或 3.11。

### 2. 一键初始化

Windows：

```powershell
.\setup_env.bat
```

Linux/macOS：

```bash
python scripts/setup_environment.py
```

脚本会按需执行以下检查：

- 如果没有 `.venv`，创建项目本地虚拟环境。
- 如果缺少依赖，安装 `requirements.txt`。
- 如果没有 `config.yaml`，从 `config.example.yaml` 复制。
- 检查 `ffmpeg` 是否可用。
- 检查项目 `_model_cache` 和全局 Hugging Face 缓存。
- 缺少 WhisperX 模型时，优先下载到项目 `_model_cache`。
- 默认不下载 Qwen3 模型；需要时添加 `--with-qwen`。

常用参数：

```powershell
.\setup_env.bat --skip-models
.\setup_env.bat --with-qwen
.\setup_env.bat --force-install
```

### 3. 手动安装

如果不使用一键脚本，也可以手动创建环境。

Windows：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Linux/macOS：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux/macOS：

```bash
python -m pip install -r requirements.txt
```

如果只需要解析 PDF/DOCX 参考资料，相关依赖已经包含在 `requirements.txt` 中。

### 4. 准备 FFmpeg

项目需要 `ffmpeg` 处理音视频。确认命令可用：

```powershell
ffmpeg -version
```

### 5. 配置模型和 API

复制配置模板：

```powershell
Copy-Item config.example.yaml config.yaml
```

Linux/macOS：

```bash
cp config.example.yaml config.yaml
```

编辑 `config.yaml`：

```yaml
api:
  key: "YOUR_API_KEY"
  base_url: "YOUR_API_BASE_URL"
  model: "YOUR_MODEL"
  llm_support_json: false

asr:
  backend: "whisperx"

subtitle_rag:
  max_chars: 17
  window_seconds: 600
  overlap_seconds: 30
  max_concurrent_llm_tasks: 3
  global_analysis_enabled: true
```

常用配置：

- `api.base_url`：LLM API 地址。
- `api.key`：LLM API key。
- `api.model`：内容校对和边界规划使用的模型。
- `api.llm_support_json`：当前 API 是否支持 OpenAI JSON response format。
- `asr.backend`：默认 `whisperx`，可改为 `qwen3`。
- `demucs`：是否启用人声分离。
- `model_dir`：本地模型缓存目录，默认 `./_model_cache`。
- `subtitle_rag.max_chars`：单条字幕默认最大长度。
- `subtitle_rag.window_seconds`：LLM 分块核心窗口秒数。
- `subtitle_rag.overlap_seconds`：LLM 分块交叠秒数。
- `subtitle_rag.max_concurrent_llm_tasks`：LLM 分词和校对的最大并发数。
- `subtitle_rag.global_analysis_enabled`：是否在内容校对前生成完整转录诊断资料，默认开启。

### 6. 准备本地模型

默认 ASR 后端是 WhisperX。模型会使用 `model_dir` 指定的目录作为缓存目录。首次运行可能需要下载 WhisperX 模型和对齐模型。

如果使用 Qwen3-ASR，请把模型放到 `config.yaml` 中配置的路径：

```yaml
asr:
  backend: "qwen3"
  qwen3:
    model_path: "./_model_cache/Qwen3-ASR-1.7B"
    forced_aligner_path: "./_model_cache/Qwen3-ForcedAligner-0.6B"
```

### 7. 环境烟测

Windows：

```powershell
@'
from core.utils.config_utils import load_key
print("model_dir:", load_key("model_dir"))
print("api_model:", load_key("api.model"))
import streamlit, pandas, openpyxl, whisperx, pypdf, docx
print("imports_ok")
'@ | .\.venv\Scripts\python.exe -
```

Linux/macOS：

```bash
python - <<'PY'
from core.utils.config_utils import load_key
print("model_dir:", load_key("model_dir"))
print("api_model:", load_key("api.model"))
import streamlit, pandas, openpyxl, whisperx, pypdf, docx
print("imports_ok")
PY
```

## 如何使用

### 前端启动

Windows：

```powershell
.\run_app.bat 8504
```

或直接运行：

```powershell
.\.venv\Scripts\python.exe -m streamlit run subtitle_rag/app.py --server.port 8504
```

Linux/macOS：

```bash
streamlit run subtitle_rag/app.py --server.port 8504
```

浏览器打开：

```text
http://localhost:8504/
```

### 前端操作

1. 上传音频或视频文件。
2. 可选上传自定义词汇表，支持 `csv/xlsx/xls`。
3. 可选上传参考资料，支持 `txt/md/srt/vtt/csv/xlsx/xls/pdf/docx`。
4. 在设置中配置 LLM base URL、API key、模型、字幕长度、窗口时间和并发数。
5. 点击开始生成字幕。
6. 下载结果文件。

### 输出文件

每次任务会写入：

```text
subtitle_rag/runs/<timestamp>/
```

主要文件：

- `draft.srt`：本地规则和边界规划生成的初稿字幕。
- `final.srt`：LLM 内容校对后的最终字幕。
- `global_asr_analysis.md`：完整转录诊断资料，供人工和 agent 查看。
- `global_asr_analysis.json`：完整转录诊断资料的机器可读版本。
- `uncertain_terms.csv`：仍需人工确认的词和候选修正。
- `llm_patches.json`：LLM 返回的 patch 原始记录。
- `patch_report.csv`：patch 应用成功、失败和跳过原因。
- `cleaned_subtitles.xlsx`：字幕表格。
- `run_manifest.json`：本次任务参数和统计。
- `subtitle_rag_result.zip`：完整结果包。

## Agent 调用说明

不需要前端时，可以直接调用 `subtitle_rag.pipeline.process_media()`。

要求：

- 从项目根目录运行。
- 确保 `config.yaml` 已存在。
- 每次只运行一个任务；任务会清理并重建项目根目录下的 `output/`。
- 输入文件可以放在任意位置。
- 结果会写入 `subtitle_rag/runs/<timestamp>/`。

### 最小调用

Windows PowerShell：

```powershell
@'
from pathlib import Path
from subtitle_rag.pipeline import process_media

def progress(stage, value):
    print(f"{value:.0%} {stage}", flush=True)

result = process_media(
    input_path=Path("samples/input.mp4"),
    glossary_paths=[],
    reference_paths=[],
    max_chars=17,
    max_concurrent_llm_tasks=3,
    progress=progress,
)

for key, value in result.items():
    print(f"{key}: {value}")
'@ | .\.venv\Scripts\python.exe -
```

Linux/macOS：

```bash
python - <<'PY'
from pathlib import Path
from subtitle_rag.pipeline import process_media

def progress(stage, value):
    print(f"{value:.0%} {stage}", flush=True)

result = process_media(
    input_path=Path("samples/input.mp4"),
    glossary_paths=[],
    reference_paths=[],
    max_chars=17,
    max_concurrent_llm_tasks=3,
    progress=progress,
)

for key, value in result.items():
    print(f"{key}: {value}")
PY
```

### 带词汇表和参考资料

```python
from subtitle_rag.pipeline import process_media

result = process_media(
    input_path="samples/input.mp4",
    glossary_paths=[
        "samples/glossary.xlsx",
        "samples/terms.csv",
    ],
    reference_paths=[
        "samples/reference.pdf",
        "samples/notes.docx",
        "samples/script.txt",
    ],
    max_chars=17,
    window_seconds=600,
    overlap_seconds=30,
    max_concurrent_llm_tasks=3,
)
```

### 词汇表格式

词汇表可以带表头，也可以不带表头。读取规则：

- 如果第一行是 `alias/canonical/note`、`术语/正确写法/备注` 等表头，会自动跳过表头。
- 如果第一行就是词条，会从第一行开始读取。
- 默认第 1 列是 ASR 可能识别出的写法，第 2 列是推荐写法，第 3 列是备注。

示例：

```csv
墨子沙龙,墨子沙龙
杨诗霞,杨石霞
```

或：

```csv
alias,canonical,note
莫子沙龙,墨子沙龙,品牌名
```

### 只跑 ASR 检查

用于检查 ASR 原始输出和词级时间戳，不进入 LLM 分词和校对：

```powershell
.\.venv\Scripts\python.exe scripts\run_asr_only.py samples\input.mp4
```

输出会包含 `cleaned_chunks.xlsx` 和 Qwen 后端的 `qwen_raw_results.jsonl`。WhisperX 后端不会生成 Qwen raw 文件。

### 从已有 ASR 结果重跑后半段

如果已经有 `output/log/cleaned_chunks.xlsx`，可以跳过 ASR，只重跑全局诊断、LLM 分词、draft 生成和内容校对：

```powershell
.\.venv\Scripts\python.exe scripts\rerun_from_last_asr.py --asr-xlsx output\log\cleaned_chunks.xlsx --input-path samples\input.mp4
```

可追加词汇表、参考资料和并发参数：

```powershell
.\.venv\Scripts\python.exe scripts\rerun_from_last_asr.py `
  --asr-xlsx output\log\cleaned_chunks.xlsx `
  --input-path samples\input.mp4 `
  --glossary samples\glossary.xlsx `
  --reference samples\notes.docx `
  --max-concurrent-llm-tasks 3
```

## 开发检查

语法检查：

```powershell
.\.venv\Scripts\python.exe -m py_compile subtitle_rag\app.py subtitle_rag\pipeline.py subtitle_rag\global_analysis.py subtitle_rag\patching.py subtitle_rag\planning.py core\_2_asr.py core\asr_backend\whisperX_local.py scripts\rerun_from_last_asr.py
```

当前主流程依赖共享的 `output/` 临时目录，因此不支持多个完整媒体任务同时运行。`subtitle_rag.max_concurrent_llm_tasks` 只控制单个任务内部的 LLM 分块并发。
