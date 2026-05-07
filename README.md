# ASR-MREnderman

从 VideoLingo 项目中摘出的本地字幕生成与 LLM 校对工具。这个目录可以独立运行，不走原项目的翻译和配音流程。

## 功能

- 本地上传音频或视频。
- 复用 WhisperX 本地识别、Demucs 人声分离、词级时间戳和现有 LLM 配置。
- 先生成 `draft.srt`，再按配置的滑动窗口加交叠交给 LLM 审稿。
- 上传的参考资料会在内容校对阶段完整提供给 LLM，用于术语、人名、机构名和专业概念纠错。
- LLM 只返回需要修改的 patch，程序应用后生成 `final.srt`。
- 输出待确认词、patch 报告和完整 zip。

## 目录

- `subtitle_rag/`：前端和字幕处理主流程。
- `core/`：从 VideoLingo 复制过来的 ASR、Demucs、LLM 调用和配置工具。
- `_model_cache/`：已复制的 WhisperX / 对齐模型缓存。
- `.venv/`：已复制的 Python 环境。
- `config.yaml`：模型、WhisperX、Demucs、LLM API 配置。
- `run_app.bat`：启动脚本。

## 启动

双击 `run_app.bat`，默认端口是 `8504`。

也可以在 PowerShell 中运行：

```powershell
Set-Location H:\ASR-MREnderman
.\run_app.bat 8504
```

浏览器打开：

```text
http://localhost:8504/
```

如果想使用 8503：

```powershell
.\run_app.bat 8503
```

## Agent 直接调用

如果不需要前端，agent 可以直接在项目根目录调用 `subtitle_rag.pipeline.process_media()`。必须从 `H:\ASR-MREnderman` 运行，确保 `config.yaml`、`output/` 和相对模型目录都指向当前项目。

最小示例：

```powershell
Set-Location H:\ASR-MREnderman
@'
from pathlib import Path
from subtitle_rag.pipeline import process_media

def progress(stage, value):
    print(f"{value:.0%} {stage}", flush=True)

result = process_media(
    input_path=Path(r"H:\path\to\input.mp4"),
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

带词汇表和参考资料：

```python
result = process_media(
    input_path=r"H:\path\to\input.mp4",
    glossary_paths=[
        r"H:\path\to\glossary.xlsx",
        r"H:\path\to\terms.csv",
    ],
    reference_paths=[
        r"H:\path\to\reference.pdf",
        r"H:\path\to\notes.docx",
        r"H:\path\to\script.txt",
    ],
    max_chars=17,
)
```

Agent 操作约定：

- 优先使用 `.\.venv\Scripts\python.exe`，不要误用系统 Python。
- 每次只跑一个任务；底层会清理并重建项目根目录下的 `output/`，不支持并发。
- 输入文件可以放在任意位置，运行结果会归档到 `subtitle_rag\runs\<timestamp>\`。
- `process_media()` 返回的 `final_srt` 是最终字幕；如果 LLM patch 失败，检查 `draft_srt`、`patch_report.csv` 和 `run_manifest.json`。
- 字幕最大长度由 `max_chars` 控制，默认 17；LLM patch 应用阶段允许额外放宽 5 个非空白字符。
- `max_concurrent_llm_tasks` 控制 LLM 分词规划和 LLM 内容校对的并发通道，默认 3；底层 ASR 和整个任务本身仍然一次只跑一个。
- 如果只想检查环境，不要启动前端，可以运行下面的烟测。

环境烟测：

```powershell
Set-Location H:\ASR-MREnderman
@'
from core.utils.config_utils import load_key
print("model_dir:", load_key("model_dir"))
print("api_model:", load_key("api.model"))
import streamlit, pandas, openpyxl, whisperx, pypdf, docx
print("imports_ok")
'@ | .\.venv\Scripts\python.exe -
```

代码编译检查：

```powershell
Set-Location H:\ASR-MREnderman
.\.venv\Scripts\python.exe -m py_compile subtitle_rag\app.py subtitle_rag\pipeline.py subtitle_rag\patching.py subtitle_rag\subtitle.py core\_2_asr.py core\asr_backend\whisperX_local.py
```

## 输出

每次任务会写入：

```text
subtitle_rag\runs\<timestamp>\
```

主要文件：

- `draft.srt`：本地规则生成的初稿字幕。
- `final.srt`：LLM patch 校对后的最终字幕。
- `uncertain_terms.csv`：仍需人工确认或由 suggested_text 替换过的词。
- `llm_patches.json`：LLM 返回的补丁原始记录。
- `patch_report.csv`：patch 应用成功、失败和跳过原因。
- `cleaned_subtitles.xlsx`：最终字幕表格。
- `run_manifest.json`：本次任务参数和统计。

## 配置

常用配置在 `config.yaml`：

- `api.base_url`
- `api.model`
- `api.key`
- `demucs`
- `whisper.runtime`
- `model_dir`
- `subtitle_rag.max_chars`
- `subtitle_rag.window_seconds`
- `subtitle_rag.overlap_seconds`
- `subtitle_rag.max_concurrent_llm_tasks`

当前 `model_dir` 是相对路径 `./_model_cache`，因此在本目录启动时会使用 `H:\ASR-MREnderman\_model_cache`。

## 说明

- 首版仍保留 VideoLingo 的 `core/` 模块作为共享轮子，避免重写 WhisperX、Demucs 和 LLM 调用。
- `output/` 是 ASR 临时工作区，每次处理会清理并重建。
- 不支持多个任务并发处理，因为底层临时目录仍是共享的 `output/`。
