@echo off
title Industrial RAG MVP

echo ============================================
echo   Industrial RAG MVP - Quick Start
echo ============================================
echo.

REM Use embedded Python (no install required)
set PYTHON=python\python.exe
set PIP=python\python.exe -m pip

REM Check embedded Python exists
if not exist "%PYTHON%" (
    echo [ERROR] Embedded Python not found in python/ directory
    pause
    exit /b 1
)

REM Install dependencies from local packages (offline, no internet needed)
echo [1/2] Installing dependencies...
%PIP% install --no-index --find-links=packages -r requirements.txt -q 2>nul

REM Start Streamlit
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
