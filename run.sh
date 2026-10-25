#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="$DIR/src:$PYTHONPATH"

echo "========================================================"
echo "  Starting CORTEX-M Studio...                           "
echo "========================================================"
python3 -m cortex_m.server.app
