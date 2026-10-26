@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m notebook --notebook-dir .
) else if exist "..\.venv\Scripts\python.exe" (
  "..\.venv\Scripts\python.exe" -m notebook --notebook-dir .
) else (
  python -m notebook --notebook-dir .
)
if errorlevel 1 pause
