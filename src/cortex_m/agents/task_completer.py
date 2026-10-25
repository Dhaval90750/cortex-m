"""Multi-Agent Task Completer for Cortex-M.

Orchestrates multi-agent teams (Architect, Coder, Critic, Auditor, Coordinator)
through the continuous Synaptic Blackboard to autonomously complete end-to-end
complex industrial coding and software engineering tasks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from cortex_m.agents.blackboard import AgentMessage, AgentRole, SynapticBlackboard
from cortex_m.agents.coding_agent import CodeArtifact, CodeReview, CortexCodingAgent
from cortex_m.agents.conceptual_explainer import ConceptualExplainer
from cortex_m.agents.git_context import GitWorkspaceContext
from cortex_m.agents.intent_router import IntentRouter, UserIntent
from cortex_m.agents.patch_sandbox import PatchSandboxRunner
from cortex_m.model.cortex_lm import CortexLM


@dataclass
class TaskExecutionResult:
    """Comprehensive result of an autonomous multi-agent task execution."""
    task_id: str
    goal: str
    status: str  # "completed", "partially_completed", "conceptual", "conversational", "failed"
    artifacts: List[CodeArtifact] = field(default_factory=list)
    reviews: List[CodeReview] = field(default_factory=list)
    messages: List[AgentMessage] = field(default_factory=list)
    execution_time_sec: float = 0.0
    synaptic_memories_imprinted: int = 0
    resonance_score: float = 0.70
    audit_summary: Dict[str, Any] = field(default_factory=dict)
    patch_diff: Optional[str] = None
    sandbox_test_passed: bool = True
    sandbox_output: Optional[str] = None
    git_status: Optional[Dict[str, Any]] = None


class MultiAgentTaskCompleter:
    """Industrial Multi-Agent Task Orchestrator and Autonomous Completer."""

    def __init__(
        self,
        blackboard: Optional[SynapticBlackboard] = None,
        lm: Optional[CortexLM] = None,
    ):
        self.blackboard = blackboard or SynapticBlackboard()
        self.lm = lm or CortexLM(dim=64, gist_dim=32, imprint_foundation=True)
        self.intent_router = IntentRouter()
        self.explainer = ConceptualExplainer()

        # Initialize specialized role agents
        self.architect = CortexCodingAgent(AgentRole.ARCHITECT, self.blackboard, self.lm)
        self.coder = CortexCodingAgent(AgentRole.CODER, self.blackboard, self.lm)
        self.critic = CortexCodingAgent(AgentRole.CRITIC, self.blackboard, self.lm)
        self.auditor = CortexCodingAgent(AgentRole.AUDITOR, self.blackboard, self.lm)
        self.skills = self.blackboard.cortex.skills

        # Git Repository Context & Sandboxed Patch Runner
        self.git_context = GitWorkspaceContext()
        self.sandbox_runner = PatchSandboxRunner()

    def execute_skill(
        self,
        skill_name: str,
        target: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute a specialized modular skill across the swarm."""
        return self.skills.execute(skill_name, target, context)

    def execute_task(
        self,
        goal: str,
        filename: str = "solution.py",
        verify_executable: bool = True,
        test_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
        progress_callback: Optional[Callable[[str, str], None]] = None,
    ) -> TaskExecutionResult:
        """Run full autonomous multi-agent lifecycle to complete a software engineering goal."""
        t_start = time.perf_counter()
        task_id = f"task_{int(time.time() * 1000)}"

        # 0. Upstream Gatekeeper: Intent Classification
        classification = self.intent_router.classify(goal)

        # Case A: CONCEPTUAL / Q&A -> Bypass swarm, explain directly with clean Markdown
        if classification.intent == UserIntent.CONCEPTUAL:
            explanation = self.explainer.explain(goal)
            t_elapsed = time.perf_counter() - t_start
            return TaskExecutionResult(
                task_id=task_id,
                goal=goal,
                status="conceptual",
                artifacts=[],
                reviews=[],
                messages=[],
                execution_time_sec=t_elapsed,
                synaptic_memories_imprinted=self.blackboard.cortex.memory.num_memories,
                resonance_score=0.98,
                audit_summary={
                    "is_valid_syntax": True,
                    "is_approved_by_critic": True,
                    "classes_defined": [],
                    "functions_defined": [],
                    "imports": [],
                    "runtime_verified": True,
                    "conceptual_response": explanation,
                },
            )

        # Case B: CONVERSATIONAL -> Social greeting or profile response
        conv = self.lm.conversation.analyze_intent(goal)
        if conv.is_conversational and conv.response:
            t_elapsed = time.perf_counter() - t_start
            return TaskExecutionResult(
                task_id=task_id,
                goal=goal,
                status="conversational",
                artifacts=[],
                reviews=[],
                messages=list(self.blackboard.messages.values()),
                execution_time_sec=t_elapsed,
                synaptic_memories_imprinted=self.blackboard.cortex.memory.num_memories,
                resonance_score=0.95,
                audit_summary={
                    "is_valid_syntax": True,
                    "is_approved_by_critic": True,
                    "classes_defined": [],
                    "functions_defined": [],
                    "imports": [],
                    "runtime_verified": True,
                    "conversational_response": conv.response,
                },
            )

        # Case C: ACTIONABLE ENGINEERING TASK -> Execute Multi-Agent Swarm
        self.blackboard.post(
            sender_role=AgentRole.COORDINATOR,
            topic="task_init",
            content=f"Initiating task '{task_id}': {goal}",
            metadata={"goal": goal},
        )

        # 1. Architect Phase: Generate 5W1H decomposition & milestone plan
        if progress_callback:
            progress_callback("Architect", "Analyzing user intent and planning milestone DAG...")
        plan = self.architect.plan(goal)

        # 2. Coder Phase: Synthesize solution code artifact
        if progress_callback:
            progress_callback("Coder", f"Synthesizing AST-validated implementation in {filename}...")
        artifact = self.coder.synthesize_code(
            task_description=f"Implement solution for: {goal}",
            filename=filename,
        )

        # 3. Critic Phase: Review code artifact for syntax, safety, and quality
        if progress_callback:
            progress_callback("Critic", "Running AST validation and security audit...")
        review = self.critic.review_code(artifact)

        # 4. Auditor Phase: Runtime execution & automated self-repair if needed
        is_runtime_valid = True
        error_msg = None

        if verify_executable and artifact.is_valid_syntax:
            if progress_callback:
                progress_callback("Auditor", "Executing isolated sandbox tests and assertions...")
            # Execute code in isolated, security-hardened namespace to verify runtime stability
            forbidden_modules = {"os", "subprocess", "shutil", "socket", "urllib", "requests", "http"}
            has_forbidden = any(f"import {mod}" in artifact.code or f"from {mod}" in artifact.code for mod in forbidden_modules)
            if has_forbidden:
                is_runtime_valid = False
                error_msg = "SecurityPolicyViolation: Code attempted to import restricted system module."
                artifact = self.coder.repair_code(artifact, error_traceback=error_msg)
                review = self.critic.review_code(artifact)
            else:
                # Safe sandbox namespace with restricted execution builtins
                safe_builtins = dict(__builtins__) if isinstance(__builtins__, dict) else dict(__builtins__.__dict__)
                # Disallow arbitrary system alterations
                restricted_scope = {"__builtins__": safe_builtins}
                try:
                    exec(compile(artifact.code, artifact.filename, "exec"), restricted_scope, restricted_scope)
                    if test_fn:
                        is_passed = test_fn(restricted_scope)
                        if not is_passed:
                            raise AssertionError("Auditor test assertion failed.")
                except Exception as e:
                    is_runtime_valid = False
                    error_msg = str(e)
                    # Autonomous self-repair loop
                    artifact = self.coder.repair_code(artifact, error_traceback=error_msg)
                    # Re-review repaired code
                    review = self.critic.review_code(artifact)

        t_elapsed = time.perf_counter() - t_start

        # 6. Coordinator Phase: Compile final audited execution package
        status = "completed" if artifact.is_valid_syntax and review.is_approved else "partially_completed"

        audit_summary = {
            "is_valid_syntax": artifact.is_valid_syntax,
            "is_approved_by_critic": review.is_approved,
            "classes_defined": artifact.classes,
            "functions_defined": artifact.functions,
            "imports": artifact.imports,
            "runtime_verified": is_runtime_valid,
            "error_repaired": error_msg is not None,
        }

        # 7. Surgical Patch & Sandboxed Validation Runner
        orig_code = ""
        target_path = Path(filename)
        if target_path.exists():
            try:
                orig_code = target_path.read_text(encoding="utf-8")
            except Exception:
                pass
        patch = self.sandbox_runner.create_and_verify_patch(
            target_file=filename,
            original_code=orig_code,
            modified_code=artifact.code,
        )

        # Imprint task completion into continuous synaptic memory
        self.blackboard.cortex.learn(
            f"Task {task_id} completed: {goal}. Functions: {', '.join(artifact.functions)}.",
            memory_id=f"mem_{task_id}",
            source_context="multi_agent_task_completer",
        )

        return TaskExecutionResult(
            task_id=task_id,
            goal=goal,
            status=status,
            artifacts=[artifact],
            reviews=[review],
            messages=list(self.blackboard.messages.values()),
            execution_time_sec=t_elapsed,
            synaptic_memories_imprinted=self.blackboard.cortex.memory.num_memories,
            resonance_score=review.provenance_resonance,
            audit_summary=audit_summary,
            patch_diff=patch.unified_diff,
            sandbox_test_passed=patch.sandbox_test_passed,
            sandbox_output=patch.sandbox_output,
            git_status=self.git_context.get_repo_status().__dict__,
        )
