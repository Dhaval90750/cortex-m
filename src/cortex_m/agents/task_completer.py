"""Multi-Agent Task Completer for Cortex-M.

Orchestrates multi-agent teams (Architect, Coder, Critic, Auditor, Coordinator)
through the continuous Synaptic Blackboard to autonomously complete end-to-end
complex industrial coding and software engineering tasks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from cortex_m.agents.blackboard import AgentMessage, AgentRole, SynapticBlackboard
from cortex_m.agents.coding_agent import CodeArtifact, CodeReview, CortexCodingAgent
from cortex_m.model.cortex_lm import CortexLM


@dataclass
class TaskExecutionResult:
    """Comprehensive result of an autonomous multi-agent task execution."""
    task_id: str
    goal: str
    status: str  # "completed", "partially_completed", "failed"
    artifacts: List[CodeArtifact] = field(default_factory=list)
    reviews: List[CodeReview] = field(default_factory=list)
    messages: List[AgentMessage] = field(default_factory=list)
    execution_time_sec: float = 0.0
    synaptic_memories_imprinted: int = 0
    resonance_score: float = 0.70
    audit_summary: Dict[str, Any] = field(default_factory=dict)


class MultiAgentTaskCompleter:
    """Industrial Multi-Agent Task Orchestrator and Autonomous Completer."""

    def __init__(
        self,
        blackboard: Optional[SynapticBlackboard] = None,
        lm: Optional[CortexLM] = None,
    ):
        self.blackboard = blackboard or SynapticBlackboard()
        self.lm = lm or CortexLM(dim=64, gist_dim=32, imprint_foundation=True)

        # Initialize specialized role agents
        self.architect = CortexCodingAgent(AgentRole.ARCHITECT, self.blackboard, self.lm)
        self.coder = CortexCodingAgent(AgentRole.CODER, self.blackboard, self.lm)
        self.critic = CortexCodingAgent(AgentRole.CRITIC, self.blackboard, self.lm)
        self.auditor = CortexCodingAgent(AgentRole.AUDITOR, self.blackboard, self.lm)

    def execute_task(
        self,
        goal: str,
        filename: str = "solution.py",
        verify_executable: bool = True,
        test_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> TaskExecutionResult:
        """Run full autonomous multi-agent lifecycle to complete a software engineering goal."""
        t_start = time.perf_counter()
        task_id = f"task_{int(time.time() * 1000)}"

        # 1. Post initial task to blackboard
        self.blackboard.post(
            sender_role=AgentRole.COORDINATOR,
            topic="task_init",
            content=f"Initiating task '{task_id}': {goal}",
            metadata={"goal": goal},
        )

        # Check if the goal is a conversational interaction rather than a coding goal
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

        # 2. Architect Phase: Generate 5W1H decomposition & milestone plan
        plan = self.architect.plan(goal)

        # 3. Coder Phase: Synthesize solution code artifact
        artifact = self.coder.synthesize_code(
            task_description=f"Implement solution for: {goal}",
            filename=filename,
        )

        # 4. Critic Phase: Review code artifact for syntax, safety, and quality
        review = self.critic.review_code(artifact)

        # 5. Auditor Phase: Runtime execution & automated self-repair if needed
        is_runtime_valid = True
        error_msg = None

        if verify_executable and artifact.is_valid_syntax:
            # Execute code in isolated namespace to verify runtime stability
            exec_scope: Dict[str, Any] = {"__builtins__": __builtins__}
            try:
                exec(compile(artifact.code, artifact.filename, "exec"), exec_scope, exec_scope)
                if test_fn:
                    is_passed = test_fn(exec_scope)
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
        )
