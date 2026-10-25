"""
CORTEX-M Headless REST API Server Runner.
Launches the headless neural runtime engine.
"""

import sys
from pathlib import Path

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cortex_m.server.app import start_cortex_server

if __name__ == "__main__":
    start_cortex_server()
