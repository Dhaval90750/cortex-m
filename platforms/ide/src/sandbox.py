"""Isolated 5.0-Second Subprocess Sandbox Runner for CORTEX-IDE.

Executes test suites and synthesized programs with:
1. Hard execution timeout (default 5.0 seconds).
2. Sanitized environment variables (strips API keys and credentials).
3. Process-level isolation and clean child termination on stall.
4. Comprehensive execution telemetry (stdout, stderr, duration_ms, exit code).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class SandboxExecutionResult:
    """Telemetry report produced by the sandbox runner."""
    command: List[str]
    success: bool
    returncode: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class SubprocessSandboxRunner:
    """Executes code blocks and commands within an isolated, monitored subprocess."""

    SENSITIVE_ENV_PATTERNS = ("KEY", "SECRET", "TOKEN", "PASS", "AUTH", "CREDENTIAL")

    def __init__(
        self,
        default_timeout_seconds: float = 5.0,
        working_directory: Optional[Union[str, Path]] = None,
    ):
        self.default_timeout_seconds = default_timeout_seconds
        self.working_directory = Path(working_directory) if working_directory else Path.cwd()

    def sanitize_environment(self) -> Dict[str, str]:
        """Produce a clean environment dictionary stripping sensitive tokens and keys."""
        clean_env = {}
        for k, v in os.environ.items():
            k_upper = k.upper()
            if any(pattern in k_upper for pattern in self.SENSITIVE_ENV_PATTERNS):
                continue
            clean_env[k] = v
        # Ensure minimal standard paths
        clean_env["PYTHONUNBUFFERED"] = "1"
        clean_env["PYTHONDONTWRITEBYTECODE"] = "1"
        return clean_env

    def run_command(
        self,
        command: List[str],
        timeout_seconds: Optional[float] = None,
        cwd: Optional[Union[str, Path]] = None,
        custom_env: Optional[Dict[str, str]] = None,
    ) -> SandboxExecutionResult:
        """Run a command inside the subprocess sandbox with hard timeout enforcement."""
        timeout = timeout_seconds or self.default_timeout_seconds
        work_dir = Path(cwd) if cwd else self.working_directory

        env = self.sanitize_environment()
        if custom_env:
            env.update(custom_env)

        start_time = time.perf_counter()
        try:
            proc = subprocess.Popen(
                command,
                cwd=str(work_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            stdout, stderr = proc.communicate(timeout=timeout)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            return SandboxExecutionResult(
                command=command,
                success=(proc.returncode == 0),
                returncode=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                duration_ms=duration_ms,
                timed_out=False,
            )

        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return SandboxExecutionResult(
                command=command,
                success=False,
                returncode=-1,
                stdout=stdout,
                stderr=stderr + f"\nSandboxTimeoutExpired: Execution exceeded {timeout}s limit.",
                duration_ms=duration_ms,
                timed_out=True,
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return SandboxExecutionResult(
                command=command,
                success=False,
                returncode=-2,
                stdout="",
                stderr=f"SandboxExecutionError: {e}",
                duration_ms=duration_ms,
                timed_out=False,
            )

    def run_python_code(
        self,
        code: str,
        timeout_seconds: Optional[float] = None,
        cwd: Optional[Union[str, Path]] = None,
        python_executable: Optional[str] = None,
    ) -> SandboxExecutionResult:
        """Execute a Python snippet safely within a temporary file in the sandbox."""
        py_bin = python_executable or sys.executable
        work_dir = Path(cwd) if cwd else self.working_directory
        custom_env = {"PYTHONPATH": str(work_dir)}
        with tempfile.NamedTemporaryFile("w", suffix=".py", dir=str(work_dir), delete=False, encoding="utf-8") as tmp:
            tmp.write(code)
            tmp_path = Path(tmp.name)

        try:
            cmd = [py_bin, str(tmp_path)]
            res = self.run_command(cmd, timeout_seconds=timeout_seconds, cwd=work_dir, custom_env=custom_env)
            return res
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
