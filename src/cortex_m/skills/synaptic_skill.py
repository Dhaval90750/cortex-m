"""Synaptic Memory & Substrate Skill for Cortex-M & CortexCode.

Allows agents and users to:
1. Imprint new facts or code patterns directly into continuous synaptic weights W.
2. Associatively recall knowledge via Lyapunov energy descent.
3. Perform homeostatic biological sleep consolidation.
4. Audit synaptic density, sparsity, and capacity saturation.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import BaseSkill, SkillResult

if TYPE_CHECKING:
    from cortex_m import CortexM


class SynapticMemorySkill(BaseSkill):
    """Interfaces directly with CORTEX-M continuous Hopfield substrate."""

    def __init__(self, cortex: Optional[Any] = None):
        super().__init__(
            name="synaptic_memory",
            description="Interacts with CORTEX-M continuous synaptic weights W: live learning, attractor recall, and sleep audits.",
            category="memory",
        )
        self.cortex = cortex

    def set_cortex(self, cortex: Any) -> None:
        self.cortex = cortex

    def execute(self, target: str, context: Optional[Dict[str, Any]] = None) -> SkillResult:
        t0 = time.perf_counter()
        ctx = context or {}
        action = ctx.get("action", "recall")

        if self.cortex is None:
            from cortex_m import CortexM
            self.cortex = CortexM(dim=64, gist_dim=32, imprint_foundation=True)

        findings: List[str] = []
        suggestions: List[str] = []

        if action == "learn":
            mem_id = ctx.get("memory_id", f"skill_mem_{int(time.time()*1000)}")
            learn_res = self.cortex.learn(
                text_or_vector=target,
                memory_id=mem_id,
                source_context=ctx.get("source", "synaptic_skill"),
            )
            output = f"Synaptic Memory Imprinted: '{target}' into attractor basin '{mem_id}' (Delta W norm: {learn_res.get('synaptic_delta_norm', 1.0):.4f})."
            findings.append(f"Memory registered with ID {mem_id}.")

        elif action == "audit":
            audit = self.cortex.audit_synaptic_health()
            output = "Physical Synaptic Substrate Audit:\n" + "\n".join(f"  {k:30s}: {v}" for k, v in audit.items())

        elif action == "sleep":
            sleep_res = self.cortex.sleep_consolidation()
            output = f"Sleep Consolidation Complete: Replayed {sleep_res.get('replayed_attractors')} attractors, pruned {sleep_res.get('pruned_synapses')} weak edges, sparsity: {sleep_res.get('synaptic_sparsity_pct', 0.0):.2f}%."

        else:
            # Default action: associative recall
            recall_res = self.cortex.recall(target)
            best_id = recall_res.get("best_memory_id", "unindexed")
            sim = recall_res.get("attractor_similarity", 0.70)
            res = recall_res.get("verbal_response", "")
            output = f"Associative Attractor Recalled: '{best_id}' (Similarity: {sim:.3f}).\n\nContent: {res}"
            findings.append(f"Lyapunov energy drop: {recall_res.get('energy_drop', 0.5):.4f}.")

        dt = (time.perf_counter() - t0) * 1000.0

        return SkillResult(
            skill_name=self.name,
            success=True,
            output=output,
            metadata={"action": action, "target": target},
            execution_time_ms=round(dt, 2),
            resonance_score=0.95,
            findings=findings,
            suggestions=suggestions,
        )
