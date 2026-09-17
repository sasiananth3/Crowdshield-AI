@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run py -3.12 scripts/bootstrap.py first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" scripts\run.py
pause
