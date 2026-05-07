@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found in PATH.
  exit /b 1
)

python scripts\setup_environment.py %*
exit /b %errorlevel%
