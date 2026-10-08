@echo off
cd /d "%~dp0"
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set CUDA_VISIBLE_DEVICES=
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.bat first.
  exit /b 1
)
if not exist "models\Toto-2.0-22m\config.json" (
  echo Missing local model. Run setup.bat / download_model.py first.
  exit /b 1
)
echo Building TotoSimple.exe (CPU, local weights)...
".venv\Scripts\python.exe" -m PyInstaller --noconfirm TotoSimple.spec
if errorlevel 1 exit /b 1
echo.
echo Done. Double-click:
echo   release\TotoSimple\TotoSimple.exe
echo Keep the _internal folder next to the exe.
pause
