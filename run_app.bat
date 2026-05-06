@echo off
setlocal
cd /d "%~dp0"

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8504"

set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "RICH_NO_LEGACY_WINDOWS=1"
set "TERM=xterm-256color"

if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv\Scripts\python.exe
  echo Copy the environment or run: python -m venv .venv
  exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run subtitle_rag\app.py --server.port %PORT% --server.headless true --browser.gatherUsageStats false
