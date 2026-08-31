@echo off
REM Keep the window open no matter what: relaunch self under cmd /k
if /i not "%~1"=="_run_" (
    cmd /k "%~f0" _run_
    exit /b
)
title Batch Translator Build
setlocal
cd /d %~dp0
if errorlevel 1 (
    echo [ERROR] Cannot enter the script folder.
    echo If you double-clicked inside the zip preview, EXTRACT the zip first!
    goto :hold
)

echo Logging to build_log.txt
echo Build started %date% %time% > build_log.txt

if not exist gui.py (
    echo.
    echo [ERROR] gui.py not found in this folder.
    echo Did you EXTRACT the zip first? Do not run inside the zip preview.
    echo Put build.bat together with gui.py and the core folder.
    echo [ERROR] gui.py not found >> build_log.txt
    goto :hold
)

echo [1/5] Checking Python...
python --version
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo Install Python 3.10-3.12 from python.org and tick "Add python.exe to PATH".
    echo [ERROR] Python not found >> build_log.txt
    goto :hold
)

echo [2/5] Creating virtual environment...
if not exist .venv (
    python -m venv .venv
)
call .venv\Scripts\activate.bat

echo [3/5] Installing dependencies (about 1-2GB, please wait)...
python -m pip install --upgrade pip
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :error
pip install transformers sentencepiece modelscope huggingface-hub faster-whisper edge-tts rapidocr-onnxruntime opencv-python imageio-ffmpeg pillow pydub numpy requests PySide6 pyinstaller static-ffmpeg
if errorlevel 1 goto :error
pip install demucs --no-deps
pip install julius einops lameenc openunmix dora-search sphn --no-deps
pip install simple-lama-inpainting arabic-reshaper python-bidi fire --no-deps
if errorlevel 1 goto :error

echo [4/5] Building exe...
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

echo [5/5] Done!
echo.
echo Output: dist\BatchTranslator.exe
echo AI models download automatically on first run, only once.
echo SUCCESS >> build_log.txt
goto :hold

:error
echo.
echo [FAILED] Build failed. Please send back this window content or build_log.txt
echo FAILED >> build_log.txt

:hold
echo.
echo ============================================================
echo  Window will stay open. Screenshot this if you need support.
echo ============================================================
