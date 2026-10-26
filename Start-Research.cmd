@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" "launch_research.py"
if errorlevel 1 pause
