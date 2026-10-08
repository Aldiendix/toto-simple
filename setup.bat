@echo off
cd /d "%~dp0"
echo Creating CPU-only venv (no NVIDIA / no CUDA)...
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo Installing CPU torch...
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python download_model.py
echo.
echo Setup done. Next:
echo   run.bat          open the widget
echo   build_exe.bat    make TotoSimple.exe
pause
