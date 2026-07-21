@echo off
title Industrial RAG MVP

echo ============================================
echo   Industrial RAG MVP - Quick Start
echo ============================================
echo.

REM Mode 1: Embedded Python (no install required, offline)
if exist "python\python.exe" (
    echo [INFO] Found embedded Python
    set PYTHON=python\python.exe
    set PIP=python\python.exe -m pip
    if exist "packages\*.whl" (
        echo [1/2] Installing dependencies (offline)...
        %PIP% install --no-index --find-links=packages -r requirements.txt -q 2>nul
    ) else (
        echo [1/2] Installing dependencies (online)...
        %PIP% install -r requirements.txt -q 2>nul
    )
    goto :start
)

REM Mode 2: System Python (requires Python installed)
echo [INFO] No embedded Python, trying system Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo.
    echo Options:
    echo   1. Install Python 3.10+ from https://www.python.org/downloads/
    echo   2. Or get the full package with embedded Python
    echo.
    pause
    exit /b 1
)
set PYTHON=python
set PIP=python -m pip

REM Check if venv exists, create if not
if not exist "venv\Scripts\activate.bat" (
    echo [1/2] Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat
echo [1/2] Installing dependencies...
pip install -r requirements.txt -q 2>nul
if errorlevel 1 (
    echo [WARN] Retrying with mirror...
    pip install -r requirements.txt -q -i https://pypi.tuna.tsinghua.edu.cn/simple
)
set PYTHON=python
set PIP=pip

:start
echo [2/2] Starting app...
echo.
echo ============================================
echo   Open browser at: http://localhost:8501
echo   Configure API Key in the sidebar
echo   Press Ctrl+C to stop
echo ============================================
echo.

%PYTHON% -m streamlit run app.py --server.port 8501

pause
