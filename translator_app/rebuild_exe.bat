@echo off
if /i not "%~1"=="_run_" (
    cmd /k "%~f0" _run_
    exit /b
)
title Batch Translator Rebuild
setlocal
cd /d %~dp0

if not exist .venv\Scripts\activate.bat (
    echo [ERROR] .venv not found. Please run build.bat first.
    goto :hold
)
call .venv\Scripts\activate.bat

echo Rebuilding exe (2-5 minutes)...
pip install imageio-ffmpeg
pip install simple-lama-inpainting arabic-reshaper python-bidi fire --no-deps
pyinstaller --noconfirm --clean --onefile --windowed ^
    --name "BatchTranslator" ^
    --collect-all faster_whisper ^
    --collect-all ctranslate2 ^
    --collect-all rapidocr_onnxruntime ^
    --collect-all imageio_ffmpeg ^
    --collect-all demucs ^
    --hidden-import sentencepiece ^
    --hidden-import modelscope ^
    gui.py
if errorlevel 1 goto :error

echo.
echo Done! New exe is at dist\BatchTranslator.exe
goto :hold

:error
echo.
echo [FAILED] Rebuild failed, screenshot this window for support.

:hold
echo.
echo Window stays open.
