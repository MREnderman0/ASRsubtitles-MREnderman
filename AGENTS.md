# AGENTS.md

## Project Defaults

- Work from `H:\ASR-MREnderman` as the project root unless the user explicitly says otherwise.
- Use `.\.venv\Scripts\python.exe` for Python commands in this project.
- Prefer calling `subtitle_rag.pipeline.process_media()` directly when the user does not need the Streamlit frontend.
- Do not run multiple subtitle jobs concurrently. The pipeline uses the shared `output\` working directory and clears it at the start of a task.

## Chinese Text And PowerShell

PowerShell inline Chinese input is not reliable on this machine. It may run without an error while silently converting Chinese characters to `?`, especially with commands like:

```powershell
@'
中文内容
'@ | .\.venv\Scripts\python.exe -
```

For tests, LLM prompt checks, JSON fixtures, SRT samples, and any command where Chinese content matters:

- Prefer reading Chinese text from an existing UTF-8 file.
- If a new fixture is needed, create a UTF-8 file first, then make Python read that file.
- For very small literals, use Python Unicode escape strings such as `\u4e2d\u6587`.
- Do not trust terminal-rendered Chinese as proof that file contents are correct; verify with UTF-8-aware file reads.
- Avoid judging LLM behavior from a PowerShell pipe that contains inline Chinese.

## Useful Checks

Environment smoke test:

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

Code compile check:

```powershell
Set-Location H:\ASR-MREnderman
.\.venv\Scripts\python.exe -m py_compile subtitle_rag\app.py subtitle_rag\pipeline.py subtitle_rag\patching.py subtitle_rag\subtitle.py core\_2_asr.py core\asr_backend\whisperX_local.py
```
