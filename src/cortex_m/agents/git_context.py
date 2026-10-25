"""Git Workspace Context & Repository Indexer for CortexCode.

Provides:
- Git repository status detection (branch, modified, untracked).
- AST-level symbol extraction across repository files (classes, functions, signatures).
- Imprints repository architecture into Cortex-M working memory for grounded code synthesis.
"""

from __future__ import annotations

import ast
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class SymbolInfo:
    name: str
    kind: str  # "class", "function", "import"
    file_path: str
    line_number: int
    docstring: Optional[str] = None
    args: List[str] = field(default_factory=list)


@dataclass
class GitRepoStatus:
    is_git_repo: bool
    branch: Optional[str] = None
    commit_hash: Optional[str] = None
    modified_files: List[str] = field(default_factory=list)
    untracked_files: List[str] = field(default_factory=list)
    total_files: int = 0


class GitWorkspaceContext:
    """Inspects and indexes the active Git repository workspace."""

    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = Path(workspace_root or os.getcwd()).resolve()
        self.symbols: Dict[str, SymbolInfo] = {}

    def get_repo_status(self) -> GitRepoStatus:
        """Inspect active git repository status."""
        git_dir = self.workspace_root / ".git"
        if not git_dir.exists():
            return GitRepoStatus(is_git_repo=False)

        try:
            # Branch
            branch_out = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                timeout=3,
            )
            branch = branch_out.stdout.strip() if branch_out.returncode == 0 else "main"

            # Commit
            commit_out = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                timeout=3,
            )
            commit = commit_out.stdout.strip() if commit_out.returncode == 0 else "unknown"

            # Status
            status_out = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                timeout=3,
            )
            modified = []
            untracked = []
            if status_out.returncode == 0:
                for line in status_out.stdout.splitlines():
                    if line.startswith(" M ") or line.startswith("M "):
                        modified.append(line[3:].strip())
                    elif line.startswith("??"):
                        untracked.append(line[3:].strip())

            return GitRepoStatus(
                is_git_repo=True,
                branch=branch,
                commit_hash=commit,
                modified_files=modified,
                untracked_files=untracked,
                total_files=len(list(self.workspace_root.rglob("*.py"))),
            )
        except Exception:
            return GitRepoStatus(is_git_repo=True, branch="main")

    def scan_symbols(self, max_files: int = 50) -> Dict[str, SymbolInfo]:
        """Scan Python files and extract class and function definitions."""
        scanned = 0
        self.symbols.clear()

        for py_path in self.workspace_root.rglob("*.py"):
            if any(part.startswith(".") or part in ("venv", ".venv", ".venv-mimo", "node_modules", "build", "dist") for part in py_path.parts):
                continue
            if scanned >= max_files:
                break
            scanned += 1

            rel_path = str(py_path.relative_to(self.workspace_root))
            try:
                with open(py_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                tree = ast.parse(content, filename=rel_path)
            except Exception:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    doc = ast.get_docstring(node)
                    self.symbols[node.name] = SymbolInfo(
                        name=node.name,
                        kind="class",
                        file_path=rel_path,
                        line_number=node.lineno,
                        docstring=doc[:120] if doc else None,
                    )
                elif isinstance(node, ast.FunctionDef):
                    doc = ast.get_docstring(node)
                    args = [a.arg for a in node.args.args]
                    self.symbols[node.name] = SymbolInfo(
                        name=node.name,
                        kind="function",
                        file_path=rel_path,
                        line_number=node.lineno,
                        docstring=doc[:120] if doc else None,
                        args=args,
                    )

        return self.symbols

    def format_architecture_summary(self) -> str:
        """Produce an architectural digest for grounding agents."""
        status = self.get_repo_status()
        symbols = self.scan_symbols()

        classes = [s.name for s in symbols.values() if s.kind == "class"][:15]
        funcs = [s.name for s in symbols.values() if s.kind == "function"][:20]

        summary = [
            f"Git Repository Context (Branch: {status.branch or 'N/A'}, Commit: {status.commit_hash or 'N/A'}):",
            f"- Workspace Root: {self.workspace_root}",
            f"- Tracked Python Symbols: {len(symbols)} definitions indexed.",
            f"- Primary Classes: {', '.join(classes) if classes else 'None'}",
            f"- Primary Functions: {', '.join(funcs) if funcs else 'None'}",
        ]
        if status.modified_files:
            summary.append(f"- Uncommitted Changes: {', '.join(status.modified_files[:5])}")

        return "\n".join(summary)
