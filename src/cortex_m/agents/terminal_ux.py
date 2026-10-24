"""Terminal UX, Single-Line Live Spinner & In-Place Status Mutator for Cortex-M Swarm.

Provides clean, modern CLI formatting:
- Single-line in-place mutating spinner with transient cleanup (no vertical scroll drift).
- Rich Markdown syntax rendering for conceptual explanations.
- Syntactic inline color diffs for code updates.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

try:
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.spinner import Spinner
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class SwarmStatus:
    """Manages a clean, mutating single-line status display for the CLI."""

    def __init__(self, initial_agent: str, initial_msg: str):
        self.agent = initial_agent
        self.msg = initial_msg

    def update(self, agent: str, msg: str):
        """Update the active agent and message safely."""
        self.agent = agent
        self.msg = msg


@contextmanager
def swarm_progress(initial_agent: str = "Architect", initial_msg: str = "Deconstructing intent...") -> Generator:
    """Single-line live mutating spinner using Rich Live with transient=True.

    Ensures that multi-agent updates (Architect -> Coder -> Critic -> Auditor) mutate
    on a single line without scrolling down the developer's terminal window, and completely
    vanishes when the task completes.
    """
    if not RICH_AVAILABLE:
        # Fallback if rich is not available
        def fallback_update(agent: str, msg: str):
            pass
        yield fallback_update
        return

    console = Console()
    status = SwarmStatus(initial_agent, initial_msg)

    # Spinner animation updates inside the Live display context
    with Live(
        Spinner("dots", text=Text.assemble(
            (f" [{status.agent}] ", "bold cyan"),
            (status.msg, "bright_black")
        )),
        console=console,
        refresh_per_second=14,
        transient=True,
    ) as live:
        def update_status(new_agent: str, new_msg: str):
            status.update(new_agent, new_msg)
            live.update(
                Spinner("dots", text=Text.assemble(
                    (f" [{status.agent}] ", "bold cyan"),
                    (status.msg, "bright_black")
                ))
            )

        yield update_status


def render_markdown(md_text: str):
    """Render structured Markdown text with terminal syntax styling."""
    if RICH_AVAILABLE:
        console = Console()
        console.print(Markdown(md_text))
    else:
        print(md_text)
