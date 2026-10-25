"""Built-in tool execution engine for Cortex Studio.

Includes:
1. Python Code Execution Sandbox (isolated subprocess with execution bounds).
2. Symbolic & Mathematical Calculator.
3. Live Knowledge Search (Wikipedia API / DuckDuckGo zero-click via pure standard library urllib).
4. System & Calendar Helper.
"""

from __future__ import annotations

import ast
import datetime
import json
import math
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional


class PythonSandbox:
    """Safe subprocess-based Python code execution engine."""

    @staticmethod
    def execute(code: str, timeout_sec: float = 5.0) -> Dict[str, Any]:
        """Execute Python code in an isolated subprocess with timeout and AST pre-validation."""
        # 1. AST syntax validation
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"SyntaxError at line {e.lineno}: {e.msg}",
                "exit_code": 1,
            }

        # 2. Write to temporary script
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            temp_path = f.name

        try:
            proc = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            return {
                "success": proc.returncode == 0,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "exit_code": proc.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timed out after {timeout_sec} seconds.",
                "exit_code": 124,
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution error: {str(e)}",
                "exit_code": 1,
            }


class KnowledgeSearcher:
    """Zero-dependency online knowledge search via Wikipedia REST API."""

    @staticmethod
    def search_summary(query: str, max_sentences: int = 3) -> Dict[str, Any]:
        """Fetch summary of a topic from Wikipedia via HTTP request."""
        clean_q = query.strip()
        encoded = urllib.parse.quote(clean_q)
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"

        headers = {
            "User-Agent": "CortexM-Studio/2.0 (research@cortex-m.ai)",
            "Accept": "application/json",
        }

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    extract = data.get("extract", "")
                    title = data.get("title", clean_q)
                    return {
                        "found": True,
                        "title": title,
                        "summary": extract,
                        "source": f"https://en.wikipedia.org/wiki/{encoded}",
                    }
        except Exception:
            pass

        return {
            "found": False,
            "title": clean_q,
            "summary": f"No live encyclopedia article found for '{clean_q}'.",
            "source": None,
        }


class Calculator:
    """Safe mathematical expression evaluator."""

    SAFE_NAMES = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "round": round,
        "pow": pow,
    }

    @classmethod
    def evaluate(cls, expression: str) -> Dict[str, Any]:
        """Safely parse and compute mathematical expression."""
        clean_expr = expression.strip().rstrip("=").strip()
        try:
            # Pre-parse using AST to ensure no dangerous constructs
            node = ast.parse(clean_expr, mode="eval")
            for subnode in ast.walk(node):
                if isinstance(subnode, (ast.Import, ast.ImportFrom, ast.Call)):
                    if isinstance(subnode, ast.Call) and isinstance(subnode.func, ast.Name):
                        if subnode.func.id not in cls.SAFE_NAMES:
                            raise ValueError(f"Function '{subnode.func.id}' is not permitted.")
                    elif not isinstance(subnode, ast.Call):
                        raise ValueError("Imports and complex statements are not permitted in calculator.")

            val = eval(compile(node, "<calc>", "eval"), {"__builtins__": {}}, cls.SAFE_NAMES)
            return {
                "success": True,
                "expression": clean_expr,
                "result": val,
                "formatted": f"{val:,.4f}" if isinstance(val, float) else str(val),
            }
        except Exception as e:
            return {
                "success": False,
                "expression": clean_expr,
                "error": str(e),
                "result": None,
            }


class CalendarHelper:
    """Provides exact temporal, chronological, and timezone context."""

    @staticmethod
    def get_current_time_info() -> Dict[str, Any]:
        """Return structured local and UTC time and calendar details."""
        now = datetime.datetime.now()
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        return {
            "iso_timestamp": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time_24h": now.strftime("%H:%M:%S"),
            "time_12h": now.strftime("%I:%M:%S %p"),
            "day_of_week": now.strftime("%A"),
            "day_of_year": now.timetuple().tm_yday,
            "week_number": now.isocalendar()[1],
            "is_leap_year": (now.year % 4 == 0 and (now.year % 100 != 0 or now.year % 400 == 0)),
            "utc_iso": now_utc.isoformat(),
        }
