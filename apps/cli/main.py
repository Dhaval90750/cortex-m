"""
CORTEX-M Universal CLI Runner.
Unified command-line interface entry point.
"""

import sys
from pathlib import Path

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cortex_m.cli import cortex_unified_main

if __name__ == "__main__":
    cortex_unified_main()
