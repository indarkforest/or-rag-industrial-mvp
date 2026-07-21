#!/bin/bash
set -e

echo "============================================"
echo "  Industrial RAG MVP - Quick Start"
echo "============================================"
echo

# Mode 1: Embedded Python
if [ -f "python/bin/python3" ]; then
    echo "[INFO] Found embedded Python"
    PYTHON="python/bin/python3"
    if ls packages/*.whl 1>/dev/null 2>&1; then
        echo "[1/2] Installing dependencies (offline)..."
        $PYTHON -m pip install --no-index --find-links=packages -r requirements.txt -q
    else
        echo "[1/2] Installing dependencies (online)..."
        $PYTHON -m pip install -r requirements.txt -q
    fi
else
    # Mode 2: System Python
    echo "[INFO] No embedded Python, trying system Python..."
    if ! command -v python3 &> /dev/null; then
        echo "[ERROR] Python3 not found!"
        echo "Options:"
        echo "  1. Install Python 3.10+"
        echo "  2. Or get the full package with embedded Python"
        exit 1
    fi
    if [ ! -d "venv" ]; then
        echo "[1/2] Creating virtual environment..."
        python3 -m venv venv
    fi
    source venv/bin/activate
    echo "[1/2] Installing dependencies..."
    pip install -r requirements.txt -q
    PYTHON=python
fi

echo "[2/2] Starting app..."
echo
echo "============================================"
echo "  Open browser at: http://localhost:8501"
echo "  Configure API Key in the sidebar"
echo "  Press Ctrl+C to stop"
echo "============================================"
echo

$PYTHON -m streamlit run app.py --server.port 8501
