"""AST Symbol Table Parser & Static Analysis for CORTEX-IDE.

Parses Python source files into rich symbol tables and call dependency graphs,
validates syntax integrity, and flags unsafe constructs.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union


@dataclass
class ASTSymbol:
    """An identified symbol in the abstract syntax tree."""
    name: str
    kind: str  # 'function', 'class', 'method', 'import'
    lineno: int
    end_lineno: int
    docstring: Optional[str]
    args: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)
    security_flags: List[str] = field(default_factory=list)


@dataclass
class ASTSymbolTable:
    """Collection of symbols and static analysis metrics for a file or module."""
    filename: str
    classes: Dict[str, ASTSymbol] = field(default_factory=dict)
    functions: Dict[str, ASTSymbol] = field(default_factory=dict)
    imports: List[str] = field(default_factory=list)
    security_issues: List[str] = field(default_factory=list)
    syntax_valid: bool = True
    error_message: Optional[str] = None


class ASTSymbolTableParser:
    """Extracts symbol tables, dependency call trees, and security flags from source code."""

    UNSAFE_CALLS = {"eval", "exec", "__import__", "compile"}

    def parse_source(self, code: str, filename: str = "<string>") -> ASTSymbolTable:
        """Parse source code string into an ASTSymbolTable."""
        try:
            tree = ast.parse(code, filename=filename)
        except SyntaxError as e:
            return ASTSymbolTable(
                filename=filename,
                syntax_valid=False,
                error_message=f"SyntaxError at line {e.lineno}: {e.msg}",
            )

        table = ASTSymbolTable(filename=filename, syntax_valid=True)
        visitor = _SymbolVisitor(table, self.UNSAFE_CALLS)
        visitor.visit(tree)
        return table

    def parse_file(self, filepath: Union[str, Path]) -> ASTSymbolTable:
        """Parse source file on disk into an ASTSymbolTable."""
        path = Path(filepath)
        if not path.exists():
            return ASTSymbolTable(
                filename=str(path),
                syntax_valid=False,
                error_message=f"File not found: {path}",
            )
        try:
            code = path.read_text(encoding="utf-8")
            return self.parse_source(code, filename=str(path))
        except Exception as e:
            return ASTSymbolTable(
                filename=str(path),
                syntax_valid=False,
                error_message=str(e),
            )


class _SymbolVisitor(ast.NodeVisitor):
    def __init__(self, table: ASTSymbolTable, unsafe_calls: Set[str]):
        self.table = table
        self.unsafe_calls = unsafe_calls
        self.current_class: Optional[str] = None

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.table.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            self.table.imports.append(f"{module}.{alias.name}")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        doc = ast.get_docstring(node)
        sym = ASTSymbol(
            name=node.name,
            kind="class",
            lineno=node.lineno,
            end_lineno=node.end_lineno or node.lineno,
            docstring=doc,
        )
        self.table.classes[node.name] = sym

        prev_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = prev_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._process_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._process_function(node)

    def _process_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> None:
        doc = ast.get_docstring(node)
        arg_names = [arg.arg for arg in node.args.args]
        kind = "method" if self.current_class else "function"
        full_name = f"{self.current_class}.{node.name}" if self.current_class else node.name

        sym = ASTSymbol(
            name=full_name,
            kind=kind,
            lineno=node.lineno,
            end_lineno=node.end_lineno or node.lineno,
            docstring=doc,
            args=arg_names,
        )

        # Inspect calls inside function
        calls = []
        flags = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func_name = None
                if isinstance(child.func, ast.Name):
                    func_name = child.func.id
                elif isinstance(child.func, ast.Attribute):
                    func_name = child.func.attr

                if func_name:
                    calls.append(func_name)
                    if func_name in self.unsafe_calls:
                        msg = f"Unsafe execution call '{func_name}' at line {child.lineno}"
                        flags.append(msg)
                        self.table.security_issues.append(msg)

            # Check for bare except: clauses
            if isinstance(child, ast.ExceptHandler) and child.type is None:
                msg = f"Bare 'except:' handler at line {child.lineno}"
                flags.append(msg)
                self.table.security_issues.append(msg)

        sym.calls = calls
        sym.security_flags = flags
        self.table.functions[full_name] = sym
        self.generic_visit(node)
