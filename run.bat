@echo off
cd /d "%~dp0"
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_HUB_DISABLE_TELEMETRY=1
set CUDA_VISIBLE_DEVICES=
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" app.py
) else if exist "dist\TotoSimple\TotoSimple.exe" (
  cd /d "%~dp0dist\TotoSimple"
  start "" TotoSimple.exe
) else (
  echo Run setup.bat first.
  pause
)
