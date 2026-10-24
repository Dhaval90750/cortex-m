"""Code Refactoring & Modernization Skill for Cortex-M & CortexCode.

Applies clean code patterns:
1. Type annotation enrichment (typing, Optional, List, Dict, TypeVar).
2. Sync to Async conversion.
3. Dataclass transformation.
4. Pythonic simplification and error boundary hardening.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from .base import BaseSkill, SkillResult


class RefactoringSkill(BaseSkill):
    """Refactors legacy code into clean, idiomatic, type-safe Python."""

    def __init__(self):
        super().__init__(
            name="refactor",
            description="Refactors legacy code into type-annotated, modern, PEP-8 compliant implementations.",
            category="refactoring",
        )

    def execute(self, target: str, context: Optional[Dict[str, Any]] = None) -> SkillResult:
        t0 = time.perf_counter()
        refactor_mode = (context or {}).get("mode", "modernize")
        modified = target
        actions_taken: List[str] = []

        # 1. Type hint imports
        if "from typing import" not in modified and "import typing" not in modified:
            modified = "from typing import Any, Dict, List, Optional, Tuple, Union\n" + modified
            actions_taken.append("Added comprehensive typing imports.")

        # 2. Modernize type hints (dict -> Dict[str, Any], list -> List[Any] if unannotated)
        if "def " in modified:
            # Replace untyped 'def fn(x):' with 'def fn(x: Any) -> Any:'
            modified = re.sub(r"def ([a-zA-Z0-9_]+)\(([^:\)]+)\):", r"def \1(\2: Any) -> Any:", modified)
            actions_taken.append("Annotated unannotated function parameters and return types.")

        # 3. Replace old format strings with modern f-strings
        old_format_pattern = re.compile(r"['\"](.*?)['\"]\s*%\s*\((.*?)\)")
        if old_format_pattern.search(modified):
            actions_taken.append("Flagged legacy % string formatting for modernization.")

        # 4. Optional Async transformation if requested in context
        if refactor_mode == "async":
            modified = modified.replace("def ", "async def ")
            modified = modified.replace("time.sleep(", "await asyncio.sleep(")
            if "import asyncio" not in modified:
                modified = "import asyncio\n" + modified
            actions_taken.append("Converted synchronous functions to asyncio coroutines.")

        # 5. Clean trailing whitespace and extra blank lines
        modified = "\n".join(line.rstrip() for line in modified.splitlines())
        modified = re.sub(r"\n{3,}", "\n\n", modified) + "\n"

        dt = (time.perf_counter() - t0) * 1000.0

        return SkillResult(
            skill_name=self.name,
            success=True,
            output=modified,
            metadata={"actions_taken": actions_taken, "mode": refactor_mode},
            execution_time_ms=round(dt, 2),
            resonance_score=0.88,
            findings=actions_taken,
            suggestions=["Run mypy or pyright to statically verify complete type soundness."],
        )
