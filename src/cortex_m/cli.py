"""Global CLI Entry Points for CORTEX-M Package.

Provides native console_scripts entry points for:
- cortex / cortexcode / cortex-code -> Autonomous Multi-Agent Coding Swarm
- cortex-substrate -> Biological Synaptic Substrate CLI (learn, recall, audit, sleep)
"""

from __future__ import annotations

import os
import sys

# Ensure src is always in sys.path
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def cortex_code_main():
    """Entry point for `cortex`, `cortexcode`, and `cortex-code`."""
    # Import scripts.cortex_code main
    _ROOT_DIR = os.path.dirname(_SRC_DIR)
    _SCRIPTS_DIR = os.path.join(_ROOT_DIR, "scripts")
    if _SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, _SCRIPTS_DIR)
    import cortex_code
    cortex_code.main()


def cortex_substrate_main():
    """Entry point for `cortex-substrate`."""
    _ROOT_DIR = os.path.dirname(_SRC_DIR)
    _SCRIPTS_DIR = os.path.join(_ROOT_DIR, "scripts")
    if _SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, _SCRIPTS_DIR)
    import cortex_cli
    cortex_cli.main()


def cortex_lm_main():
    """Entry point for `cortexlm` and `cortex-lm`."""
    _ROOT_DIR = os.path.dirname(_SRC_DIR)
    _SCRIPTS_DIR = os.path.join(_ROOT_DIR, "scripts")
    if _SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, _SCRIPTS_DIR)
    import cortex_lm_chat
    cortex_lm_chat.main()


if __name__ == "__main__":
    cortex_code_main()
