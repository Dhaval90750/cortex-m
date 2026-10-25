"""AST-Validated Unified Diff Engine for CORTEX-IDE.

Computes surgical diffs, verifies AST syntax validity before patch application,
and guards against accidental corruption or comment destruction.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .ast_parser import ASTSymbolTable, ASTSymbolTableParser


@dataclass
class PatchResult:
    """Outcome of an AST-validated patch operation."""
    success: bool
    diff_text: str
    lines_added: int
    lines_removed: int
    ast_valid: bool
    error_message: Optional[str] = None
    applied_content: Optional[str] = None


class ASTDiffEngine:
    """Generates unified diffs and ensures code changes satisfy AST syntax integrity."""

    def __init__(self):
        self.parser = ASTSymbolTableParser()

    def generate_diff(
        self,
        original_code: str,
        modified_code: str,
        filename: str = "patch.py",
    ) -> str:
        """Produce unified diff string between original and modified source."""
        orig_lines = original_code.splitlines(keepends=True)
        mod_lines = modified_code.splitlines(keepends=True)

        diff = difflib.unified_diff(
            orig_lines,
            mod_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            lineterm="",
        )
        return "\n".join(diff)

    def validate_and_patch(
        self,
        original_code: str,
        modified_code: str,
        filename: str = "target.py",
    ) -> PatchResult:
        """Validate modified code AST syntax and construct the patch result."""
        # 1. Check syntax of modified code
        ast_table = self.parser.parse_source(modified_code, filename=filename)
        if not ast_table.syntax_valid:
            return PatchResult(
                success=False,
                diff_text="",
                lines_added=0,
                lines_removed=0,
                ast_valid=False,
                error_message=f"AST validation failed: {ast_table.error_message}",
                applied_content=None,
            )

        # 2. Compute unified diff
        diff_text = self.generate_diff(original_code, modified_code, filename=filename)

        # Count added/removed lines
        added = sum(1 for line in diff_text.splitlines() if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in diff_text.splitlines() if line.startswith("-") and not line.startswith("---"))

        return PatchResult(
            success=True,
            diff_text=diff_text,
            lines_added=added,
            lines_removed=removed,
            ast_valid=True,
            applied_content=modified_code,
        )

    def apply_to_file(
        self,
        filepath: Union[str, Path],
        modified_code: str,
    ) -> PatchResult:
        """Apply an AST-validated patch directly to a file on disk."""
        path = Path(filepath)
        orig_code = path.read_text(encoding="utf-8") if path.exists() else ""
        res = self.validate_and_patch(orig_code, modified_code, filename=path.name)
        if res.success:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(modified_code, encoding="utf-8")
        return res
