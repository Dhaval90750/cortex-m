"""
Cortex Studio Application Runner.
Launches the local HTTP server and neural workbench.
"""

import sys
from pathlib import Path

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cortex_m.server.app import main

if __name__ == "__main__":
    main()
