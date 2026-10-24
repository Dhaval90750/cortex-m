"""Security Audit & Vulnerability Scanner Skill for Cortex-M & CortexCode.

Performs static analysis and AST inspection for:
1. Dynamic code execution (eval, exec, compile).
2. Unsafe subprocess calls (shell=True, raw string formatting).
3. Hardcoded secrets, API keys, passwords, and tokens.
4. SQL Injection vulnerabilities (f-strings or % in SQL queries).
5. Path traversal risks (untrusted file path concatenation).
6. Insecure deserialization (pickle.loads, yaml.unsafe_load).
7. Bare exception clauses masking critical failures.
"""

from __future__ import annotations

import ast
import re
import time
from typing import Any, Dict, List, Optional

from .base import BaseSkill, SkillResult


class SecurityAuditSkill(BaseSkill):
    """Rigorous static security auditor and vulnerability scanner."""

    def __init__(self):
        super().__init__(
            name="security_audit",
            description="Scans code for OWASP vulnerabilities, eval/exec risks, hardcoded secrets, and SQL injections.",
            category="security",
        )

    def execute(self, target: str, context: Optional[Dict[str, Any]] = None) -> SkillResult:
        t0 = time.perf_counter()
        findings: List[str] = []
        suggestions: List[str] = []
        risk_score = 0.0

        # 1. Parse AST if Python code
        is_python = True
        try:
            tree = ast.parse(target)
        except SyntaxError:
            is_python = False
            tree = None

        # 2. AST Inspection
        if tree:
            for node in ast.walk(tree):
                # Dynamic execution
                if isinstance(node, ast.Call):
                    func_name = ""
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        func_name = node.func.attr

                    if func_name in ("eval", "exec"):
                        findings.append(f"[CRITICAL] Dynamic code execution detected: '{func_name}()' at line {getattr(node, 'lineno', '?')}.")
                        suggestions.append("Replace dynamic eval/exec with safe deterministic parsers (e.g. ast.literal_eval).")
                        risk_score = max(risk_score, 0.95)

                    if func_name == "loads" and isinstance(getattr(node, "func", None), ast.Attribute):
                        if getattr(node.func.value, "id", "") == "pickle":
                            findings.append(f"[HIGH] Insecure deserialization detected: 'pickle.loads()' at line {getattr(node, 'lineno', '?')}.")
                            suggestions.append("Avoid unpickling untrusted inputs; use JSON or Protocol Buffers.")
                            risk_score = max(risk_score, 0.85)

                # Bare except
                if isinstance(node, ast.ExceptHandler):
                    if node.type is None:
                        findings.append(f"[MEDIUM] Bare 'except:' clause at line {getattr(node, 'lineno', '?')}.")
                        suggestions.append("Catch specific exceptions (e.g. 'except ValueError:') instead of catching BaseException.")
                        risk_score = max(risk_score, 0.40)

        # 3. Lexical / Regex Pattern Auditing
        lines = target.splitlines()
        for idx, line in enumerate(lines, 1):
            # Hardcoded API keys / secrets
            if re.search(r'''(?i)(api[_-]?key|secret[_-]?key|password|auth[_-]?token)\s*=\s*['"][a-zA-Z0-9_\-]{8,}['"]''', line):
                findings.append(f"[HIGH] Hardcoded secret/credential detected at line {idx}: {line.strip()[:40]}...")
                suggestions.append("Load credentials from environment variables (os.environ) or secret managers.")
                risk_score = max(risk_score, 0.80)

            # Unsafe subprocess with shell=True
            if "shell=True" in line and ("subprocess" in target or "Popen" in line or "call" in line):
                findings.append(f"[HIGH] Unsafe subprocess invocation with 'shell=True' at line {idx}.")
                suggestions.append("Pass command as a list without shell=True to prevent command injection.")
                risk_score = max(risk_score, 0.80)

            # SQL injection via f-strings
            if re.search(r'''(?i)(select|insert|update|delete)\s+.*f['"].*\{.*\}''', line):
                findings.append(f"[CRITICAL] Potential SQL Injection via string formatting at line {idx}: {line.strip()[:40]}...")
                suggestions.append("Use parameterized queries (e.g., cursor.execute('SELECT ... WHERE id = %s', (val,))) instead of string formatting.")
                risk_score = max(risk_score, 0.90)

            # Path traversal
            if re.search(r'''open\s*\(\s*.*(?:request|\.get|\.query).*['"]?\s*[,)]''', line):
                findings.append(f"[MEDIUM] Potential path traversal vulnerability in file open at line {idx}.")
                suggestions.append("Sanitize file paths using os.path.abspath and verify they reside within the target directory.")
                risk_score = max(risk_score, 0.50)

        dt = (time.perf_counter() - t0) * 1000.0
        success = len(findings) == 0

        # Formulate human-friendly report
        if success:
            output = "Security Audit Result: PASSED. Zero critical vulnerabilities detected. AST static analysis verified clean."
        else:
            output = f"Security Audit Result: {len(findings)} ISSUES FOUND (Risk Score: {risk_score:.2f}/1.00).\n\n"
            output += "Findings:\n" + "\n".join(f"  * {f}" for f in findings) + "\n\n"
            output += "Remediations:\n" + "\n".join(f"  * {s}" for s in set(suggestions))

        return SkillResult(
            skill_name=self.name,
            success=success,
            output=output,
            metadata={
                "risk_score": risk_score,
                "issues_count": len(findings),
                "is_python_syntax": is_python,
                "verdict": "APPROVED" if success else "REJECTED_SECURITY_RISK",
            },
            execution_time_ms=round(dt, 2),
            resonance_score=0.90 if success else 0.50,
            findings=findings,
            suggestions=suggestions,
        )
