@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. See docs\RUN_IN_VS_CODE.md for setup.
  exit /b 1
)
".venv\Scripts\python.exe" "scripts\check_research.py" %*
