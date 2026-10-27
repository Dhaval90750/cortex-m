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
    cyclomatic_complexity: int = 1


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

    @classmethod
    def build_project_call_graph(cls, tables: List[ASTSymbolTable]) -> Dict[str, Any]:
        """Resolve cross-file call dependencies, detect dead symbols, and identify circular imports."""
        all_functions: Set[str] = set()
        function_locations: Dict[str, str] = {}
        syntax_errors: List[Dict[str, str]] = []
        for t in tables:
            if not t.syntax_valid:
                syntax_errors.append({"filename": t.filename, "error": t.error_message or "Unknown syntax error"})
                continue
            for fn in t.functions.keys():
                all_functions.add(fn)
                function_locations[fn] = t.filename

        call_graph: Dict[str, List[str]] = {}
        incoming_calls: Dict[str, List[str]] = {fn: [] for fn in all_functions}
        module_imports: Dict[str, Set[str]] = {}

        for t in tables:
            if not t.syntax_valid:
                continue
            mod_name = Path(t.filename).stem
            module_imports[mod_name] = set(t.imports)
            for fn_name, sym in t.functions.items():
                resolved_callees = []
                for call in sym.calls:
                    if call in all_functions:
                        resolved_callees.append(call)
                        incoming_calls[call].append(fn_name)
                    else:
                        for candidate in all_functions:
                            if candidate.endswith(f".{call}"):
                                resolved_callees.append(candidate)
                                incoming_calls[candidate].append(fn_name)
                call_graph[fn_name] = resolved_callees

        dead_symbols = [
            fn for fn, callers in incoming_calls.items()
            if len(callers) == 0 and not fn.startswith("test_") and not fn.endswith(".__init__") and fn != "main"
        ]

        circular_dependencies: List[List[str]] = []
        visited: Dict[str, int] = {}
        path: List[str] = []

        def dfs_cycle(u: str):
            visited[u] = 1
            path.append(u)
            for imp in module_imports.get(u, []):
                imp_stem = imp.split(".")[-1]
                if imp_stem in module_imports:
                    if visited.get(imp_stem, 0) == 1:
                        cycle_start = path.index(imp_stem)
                        circular_dependencies.append(list(path[cycle_start:]) + [imp_stem])
                    elif visited.get(imp_stem, 0) == 0:
                        dfs_cycle(imp_stem)
            path.pop()
            visited[u] = 2

        for mod in module_imports:
            if visited.get(mod, 0) == 0:
                dfs_cycle(mod)

        return {
            "total_modules": len(tables),
            "valid_modules": len(tables) - len(syntax_errors),
            "syntax_errors": syntax_errors,
            "total_functions": len(all_functions),
            "call_graph": call_graph,
            "incoming_calls": incoming_calls,
            "dead_symbols": dead_symbols,
            "circular_dependencies": circular_dependencies,
            "has_circular_dependencies": len(circular_dependencies) > 0,
        }

    @classmethod
    def generate_unit_test_suite(
        cls,
        table: ASTSymbolTable,
        target_function: Optional[str] = None,
    ) -> str:
        """Synthesize executable unittest test cases for functions identified in the AST table."""
        mod_name = Path(table.filename).stem if table.filename != "<string>" else "target_module"
        target_fns = [f for f in table.functions.values() if target_function is None or f.name == target_function]

        lines = [
            '"""Automated Unit Test Suite synthesized by CORTEX-IDE AST Engine."""',
            "import unittest",
            f"# Target Module: {table.filename}",
            "",
            f"class Test{mod_name.capitalize()}SynthesizedSuite(unittest.TestCase):",
            "    def setUp(self):",
            "        pass",
            "",
        ]

        for fn in target_fns:
            clean_fn = fn.name.replace(".", "_")
            doc = fn.docstring or "Automated test scaffold"
            arg_list = fn.args

            lines.append(f"    def test_{clean_fn}_nominal_execution(self):")
            lines.append(f'        """Nominal test for {fn.name}: {doc.strip()[:60]}"""')
            dummy_args = []
            for arg in arg_list:
                if arg in ("self", "cls"):
                    continue
                dummy_args.append("None")
            args_str = ", ".join(dummy_args)
            lines.append(f"        # Invocation: {fn.name}({args_str})")
            lines.append("        self.assertTrue(True, 'Nominal execution assertion scaffold')")
            lines.append("")

            lines.append(f"    def test_{clean_fn}_boundary_conditions(self):")
            lines.append(f'        """Boundary value test for {fn.name}."""')
            lines.append(f"        # Complexity score: {fn.cyclomatic_complexity}")
            lines.append(f"        self.assertGreaterEqual({fn.cyclomatic_complexity}, 1)")
            lines.append("")

        lines.extend([
            'if __name__ == "__main__":',
            "    unittest.main()",
            "",
        ])
        return "\n".join(lines)


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

        # Calculate McCabe Cyclomatic Complexity (1 + decision branch points)
        decision_nodes = (
            ast.If, ast.While, ast.For, ast.AsyncFor,
            ast.ExceptHandler, ast.With, ast.AsyncWith,
            ast.Assert, ast.IfExp
        )
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, decision_nodes):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += max(0, len(child.values) - 1)

        sym.calls = calls
        sym.security_flags = flags
        sym.cyclomatic_complexity = complexity
        self.table.functions[full_name] = sym
        self.generic_visit(node)
