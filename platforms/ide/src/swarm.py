"""Autonomous Multi-Agent Engineering Swarm for CORTEX-IDE / CortexCode.

Coordinates specialized agent personas across repository tasks:
1. Architect: 5W1H DAG task decomposition and interface design.
2. Coder: Typed, docstring-annotated code synthesis.
3. Critic: AST security review and lint inspection.
4. Auditor: 5.0-second subprocess test sandbox and execution verification.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .ast_parser import ASTSymbolTable, ASTSymbolTableParser
from .diff_engine import ASTDiffEngine, PatchResult
from .sandbox import SandboxExecutionResult, SubprocessSandboxRunner


class AgentRole(str, enum.Enum):
    ARCHITECT = "Architect"
    CODER = "Coder"
    CRITIC = "Critic"
    AUDITOR = "Auditor"


@dataclass
class SwarmTaskPlan:
    """Task decomposition DAG authored by the Architect agent."""
    task_id: str
    goal: str
    target_files: List[str]
    steps: List[str]
    interface_invariants: List[str]
    created_at: float = field(default_factory=time.time)


@dataclass
class SwarmTaskResult:
    """Final outcome of an autonomous engineering swarm workflow."""
    task_id: str
    success: bool
    plan: SwarmTaskPlan
    synthesized_code: Optional[str]
    diff_result: Optional[PatchResult]
    critic_audit_passed: bool
    critic_findings: List[str]
    sandbox_result: Optional[SandboxExecutionResult]
    human_approved: bool
    duration_ms: float
    error_message: Optional[str] = None


class CortexCodeSwarm:
    """Supervises the autonomous multi-agent engineering lifecycle."""

    def __init__(
        self,
        sandbox_timeout: float = 5.0,
        working_directory: Optional[Union[str, Path]] = None,
        auto_approve_sandbox_passed: bool = False,
    ):
        self.sandbox_timeout = sandbox_timeout
        self.working_directory = Path(working_directory) if working_directory else Path.cwd()
        self.auto_approve = auto_approve_sandbox_passed

        self.ast_parser = ASTSymbolTableParser()
        self.diff_engine = ASTDiffEngine()
        self.sandbox = SubprocessSandboxRunner(
            default_timeout_seconds=sandbox_timeout,
            working_directory=self.working_directory,
        )

    def plan_task(self, goal: str, target_files: List[str]) -> SwarmTaskPlan:
        """Architect Persona: Deconstruct user task into an architectural plan."""
        task_id = f"task_{int(time.time() * 1000)}"
        steps = [
            f"1. Parse AST symbol table for target files: {', '.join(target_files)}",
            "2. Synthesize patch maintaining type signatures and comments",
            "3. Run Critic static security and syntax validation",
            "4. Execute Auditor 5.0-second subprocess sandbox test",
            "5. Await deterministic human approval before disk write",
        ]
        invariants = [
            "Syntax must be 100% valid Python AST",
            "No eval() or exec() calls allowed",
            "Execution must pass without timeout inside 5.0s sandbox",
        ]
        return SwarmTaskPlan(
            task_id=task_id,
            goal=goal,
            target_files=target_files,
            steps=steps,
            interface_invariants=invariants,
        )

    def execute_engineering_workflow(
        self,
        goal: str,
        target_file: str,
        code_synthesizer: Callable[[SwarmTaskPlan], str],
        test_script: Optional[str] = None,
        human_approval_callback: Optional[Callable[[PatchResult], bool]] = None,
    ) -> SwarmTaskResult:
        """Execute full multi-agent engineering pipeline."""
        start_time = time.perf_counter()
        plan = self.plan_task(goal, [target_file])

        # 1. Coder Persona: Synthesize patch
        try:
            synthesized_code = code_synthesizer(plan)
        except Exception as e:
            return SwarmTaskResult(
                task_id=plan.task_id,
                success=False,
                plan=plan,
                synthesized_code=None,
                diff_result=None,
                critic_audit_passed=False,
                critic_findings=[],
                sandbox_result=None,
                human_approved=False,
                duration_ms=(time.perf_counter() - start_time) * 1000.0,
                error_message=f"Coder failed during synthesis: {e}",
            )

        # 2. Critic Persona: AST security & syntax analysis
        ast_table = self.ast_parser.parse_source(synthesized_code, filename=target_file)
        if not ast_table.syntax_valid:
            return SwarmTaskResult(
                task_id=plan.task_id,
                success=False,
                plan=plan,
                synthesized_code=synthesized_code,
                diff_result=None,
                critic_audit_passed=False,
                critic_findings=[ast_table.error_message or "Invalid syntax"],
                sandbox_result=None,
                human_approved=False,
                duration_ms=(time.perf_counter() - start_time) * 1000.0,
                error_message=f"Critic rejected code: {ast_table.error_message}",
            )

        critic_passed = len(ast_table.security_issues) == 0
        if not critic_passed:
            return SwarmTaskResult(
                task_id=plan.task_id,
                success=False,
                plan=plan,
                synthesized_code=synthesized_code,
                diff_result=None,
                critic_audit_passed=False,
                critic_findings=ast_table.security_issues,
                sandbox_result=None,
                human_approved=False,
                duration_ms=(time.perf_counter() - start_time) * 1000.0,
                error_message="Critic security audit failed on unsafe constructs.",
            )

        # 3. Diff Engine: Build unified diff
        target_path = self.working_directory / target_file
        orig_code = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
        diff_res = self.diff_engine.validate_and_patch(orig_code, synthesized_code, filename=target_file)

        # 4. Auditor Persona: Run 5.0-second sandbox verification
        sandbox_res = None
        if test_script:
            sandbox_res = self.sandbox.run_python_code(test_script, timeout_seconds=self.sandbox_timeout)
            if not sandbox_res.success:
                return SwarmTaskResult(
                    task_id=plan.task_id,
                    success=False,
                    plan=plan,
                    synthesized_code=synthesized_code,
                    diff_result=diff_res,
                    critic_audit_passed=True,
                    critic_findings=[],
                    sandbox_result=sandbox_res,
                    human_approved=False,
                    duration_ms=(time.perf_counter() - start_time) * 1000.0,
                    error_message=f"Auditor test sandbox failed: {sandbox_res.stderr.strip()}",
                )

        # 5. Deterministic Human Approval Gate
        human_approved = False
        if human_approval_callback:
            human_approved = human_approval_callback(diff_res)
        elif self.auto_approve and (sandbox_res is None or sandbox_res.success):
            human_approved = True

        if human_approved:
            # Apply to disk safely
            self.diff_engine.apply_to_file(target_path, synthesized_code)

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        return SwarmTaskResult(
            task_id=plan.task_id,
            success=True,
            plan=plan,
            synthesized_code=synthesized_code,
            diff_result=diff_res,
            critic_audit_passed=True,
            critic_findings=[],
            sandbox_result=sandbox_res,
            human_approved=human_approved,
            duration_ms=duration_ms,
        )
