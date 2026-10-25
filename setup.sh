#!/usr/bin/env bash
set -e

echo "========================================================"
echo "  CORTEX-M Automated Installer (Linux / macOS)          "
echo "========================================================"
echo ""

# Find Python 3
PYTHON_BIN=""
for cmd in python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        PYTHON_BIN="$cmd"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "[ERROR] Python 3.9+ was not found on your system."
    echo "Please install Python 3 via your package manager (e.g. apt install python3 python3-venv or brew install python)."
    exit 1
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Create venv if needed
if [ ! -d ".venv" ]; then
    echo "[*] Creating Python virtual environment in .venv..."
    $PYTHON_BIN -m venv .venv
fi

# Activate venv
source .venv/bin/activate

echo "[*] Upgrading pip..."
pip install --upgrade pip

echo "[*] Installing Cortex-M in editable mode..."
pip install -e .

echo ""
echo "========================================================"
echo "  CORTEX-M Installation Complete!                       "
echo "========================================================"
echo "Run ./run.sh or 'cortex-studio' to start."
