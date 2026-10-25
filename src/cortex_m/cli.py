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


def cortex_studio_main():
    """Entry point for `cortex-studio` or `cortex studio`."""
    import argparse
    from cortex_m.server.app import start_cortex_studio
    
    parser = argparse.ArgumentParser(description="Cortex Studio Web Application Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind server (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8888, help="Port to bind server (default: 8888)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    start_cortex_studio(host=args.host, port=args.port, open_browser=not args.no_browser)


def cortex_unified_main():
    """Smart unified entry point for `cortex` command."""
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd in ("--help", "-h", "help"):
            print("""
========================================================================
 CORTEX-M: Autonomous Neural Memory & Multi-Agent Swarm Suite
========================================================================

Usage:
  cortex studio              Launch Cortex Studio Web Application (Default)
  cortex chat                Start interactive neural conversational assistant
  cortex code [prompt]       Run Autonomous Multi-Agent Coding Swarm
  cortex substrate           Biological Synaptic Memory CLI (learn, recall, sleep)
  cortex --help              Show this help message

Commands:
  cortex-studio              Direct alias to launch Studio UI
  cortex-code                Direct alias to launch Coding Swarm
  cortex-substrate           Direct alias for Synaptic Substrate

For full documentation and web interface:
  Open http://127.0.0.1:8888 after running 'cortex studio'.
""")
            return
        elif cmd == "studio":
            sys.argv.pop(1)
            cortex_studio_main()
            return
        elif cmd in ("substrate", "memory"):
            sys.argv.pop(1)
            cortex_substrate_main()
            return
        elif cmd in ("chat", "lm"):
            sys.argv.pop(1)
            cortex_lm_main()
            return

    # Default: Autonomous Multi-Agent Coding Swarm
    cortex_code_main()


if __name__ == "__main__":
    cortex_unified_main()
