@echo off
cd /d "%~dp0"
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_HUB_DISABLE_TELEMETRY=1
set CUDA_VISIBLE_DEVICES=
title Toto Simple
echo Starting Toto Simple widget...
echo.
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" app.py
  if errorlevel 1 (
    echo Widget closed with an error.
    pause
  )
  goto :eof
)
if exist "dist\TotoSimple\TotoSimple.exe" (
  cd /d "%~dp0dist\TotoSimple"
  TotoSimple.exe
  if errorlevel 1 pause
  goto :eof
)
echo Could not find the app. Run setup.bat first.
pause
