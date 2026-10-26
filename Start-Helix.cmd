@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run Setup-Helix.ps1 first with Python 3.12 installed.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" "lab\server.py"
if errorlevel 1 pause
