"""Unified Cortex-M Developer Engine & Platform SDK.

The single, high-level entry point for developers and researchers to directly
reference and use Cortex-M to create modern LLMs, neuro-symbolic agents,
and cognitive analysis systems.

Example Usage:
```python
from cortex_m.sdk import CortexEngine

# Initialize engine with 64-dim neural space and 86%+ sparsity
engine = CortexEngine()

# 1. Epistemic Learning (No Vector DB needed!)
engine.remember(
    "The quantum communication protocol runs on 1550nm fiber.",
    memory_id="quantum_comm",
    source="lab_specs",
)

# 2. Grounded Recall with Provenance Explainability
res = engine.recall("What wavelength does the quantum protocol use?")
print(res.text)  # Grounded fact
print(res.explain())  # Provenance & pathway resonance

# 3. Direct LLM Augmentation
bridge = engine.get_llm_bridge()
augmented = bridge.augment_prompt("How should I configure the optical transceiver?")

# 4. Multi-Tenant Ephemeral Isolation (2 KB per tenant)
with engine.tenant("tenant_enterprise_a") as user_a:
    user_a.remember("Confidential budget is $5M.", memory_id="budget")
    print(user_a.recall("What is the budget?").text)

# 5. Cognitive Health & Hallucination Audit
health = engine.audit()
print(health)
```
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Union

import torch

from ..cortex_v2 import CortexV2
from ..analytics.cognitive_analyzer import CognitiveAnalyzer
from .llm_bridge import CortexLLMBridge


class RecallResult:
    """Structured response container for associative recall."""

    def __init__(self, data: Dict[str, Any], analyzer: CognitiveAnalyzer):
        self._data = data
        self._analyzer = analyzer

    @property
    def text(self) -> Optional[str]:
        return self._data.get("best_metadata", {}).get("raw_text") or self._data.get("verbal_response")

    @property
    def memory_id(self) -> Optional[str]:
        return self._data.get("best_memory_id")

    @property
    def attractor_similarity(self) -> float:
        return self._data.get("attractor_similarity", 0.0)

    @property
    def resonance_score(self) -> float:
        res = self._data.get("resonance")
        return res.get("resonance_score", 0.0) if res else 0.0

    @property
    def raw(self) -> Dict[str, Any]:
        return self._data

    def explain(self) -> str:
        """Return human-readable provenance and attribution summary."""
        report = self._analyzer.generate_explainability_report(
            recall_diagnostics=self._data,
            recalled_text=self.text,
        )
        return report["summary_explanation"]

    def __repr__(self) -> str:
        return f"<RecallResult id={self.memory_id} sim={self.attractor_similarity:.3f} res={self.resonance_score:.3f}>"


class TenantSession:
    """Isolated tenant session with zero cross-tenant data leakage."""

    def __init__(self, engine: CortexEngine, tenant_id: str):
        self.engine = engine
        self.tenant_id = tenant_id

    def remember(
        self,
        text_or_vector: Union[str, Sequence[float], torch.Tensor],
        lr: float = 0.5,
    ) -> Dict[str, Any]:
        """Learn isolated memory strictly within this tenant's 2 KB adapter."""
        return self.engine.cortex.learn_for_tenant(self.tenant_id, text_or_vector, lr=lr)

    def recall(
        self,
        cue: Union[str, Sequence[float], torch.Tensor],
    ) -> RecallResult:
        """Recall with tenant-specific effective synaptic weights."""
        res = self.engine.cortex.recall_for_tenant(self.tenant_id, cue)
        return RecallResult(res, self.engine.analyzer)


class CortexEngine:
    """Unified engine for associative synaptic memory, routing, and cognitive analysis."""

    def __init__(
        self,
        dim: int = 64,
        gist_dim: int = 32,
        beta: float = 8.0,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.cortex = CortexV2(
            dim=dim,
            gist_dim=gist_dim,
            beta=beta,
            device=device,
            seed=seed,
        )
        self.analyzer = CognitiveAnalyzer(
            memory_network=self.cortex.memory,
            safety_critic=self.cortex.safety_critic,
        )
        self._llm_bridge: Optional[CortexLLMBridge] = None

    def remember(
        self,
        text_or_vector: Union[str, Sequence[float], torch.Tensor],
        memory_id: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        salience: float = 1.0,
    ) -> Dict[str, Any]:
        """Learn a new pattern directly into physical synaptic connection weights."""
        return self.cortex.learn(
            text_or_vector=text_or_vector,
            memory_id=memory_id,
            source_context=source,
            metadata=metadata,
            salience=salience,
        )

    def recall(
        self,
        cue: Union[str, Sequence[float], torch.Tensor],
        use_working_memory: bool = True,
    ) -> RecallResult:
        """Recall nearest memory via continuous Hopfield energy relaxation."""
        data = self.cortex.recall(cue, use_working_memory=use_working_memory)
        return RecallResult(data, self.analyzer)

    def forget(self, target: Union[str, torch.Tensor]) -> Dict[str, Any]:
        """Erase targeted memory via anti-Hebbian depression (Delta W = -eta v v^T)."""
        return self.cortex.forget(target)

    def route(
        self,
        cue: Union[str, torch.Tensor],
        values: Sequence[float],
        role_ids: Sequence[int],
    ) -> Dict[str, Any]:
        """Causal MIMO sparse routing conditioned on pre-verbal gist."""
        return self.cortex.route(cue, values=values, role_ids=role_ids)

    def route_hierarchical(
        self,
        cue: Union[str, torch.Tensor],
        values: Sequence[float],
        role_ids: Sequence[int],
        n_out: int = 4,
    ) -> Dict[str, Any]:
        """Hierarchical sparse routing scaling to N=16 and N=32."""
        return self.cortex.route_hierarchical(cue, values=values, role_ids=role_ids, n_out=n_out)

    def dual_stream(self, text_or_code: str) -> Dict[str, Any]:
        """Extract Ventral 128-D factored gist + Dorsal verbatim syntax/code blocks."""
        return self.cortex.process_dual_stream(text_or_code)

    def sleep(
        self,
        replay_iterations: int = 3,
        noise_level: float = 0.15,
        decay_factor: float = 0.92,
        micro_sleep_async: bool = False,
    ) -> Dict[str, Any]:
        """Run homeostatic sleep consolidation (prunes synapses to 86%+ sparsity)."""
        if micro_sleep_async:
            return self.cortex.consolidator.run_shadow_consolidation(
                replay_iterations=replay_iterations,
                noise_level=noise_level,
                decay_factor=decay_factor,
            )
        return self.cortex.sleep_consolidation(
            replay_iterations=replay_iterations,
            noise_level=noise_level,
            decay_factor=decay_factor,
        )

    @contextmanager
    def tenant(self, tenant_id: str) -> Generator[TenantSession, None, None]:
        """Context manager for isolated tenant operations (2 KB per tenant)."""
        session = TenantSession(self, tenant_id)
        yield session

    def audit(self) -> Dict[str, Any]:
        """Comprehensive audit of synaptic health, sparsity, and capacity."""
        return self.analyzer.inspect_synaptic_health()

    def get_llm_bridge(self) -> CortexLLMBridge:
        """Retrieve turnkey LLM Bridge for augmenting modern foundation models."""
        if self._llm_bridge is None:
            self._llm_bridge = CortexLLMBridge(cortex=self.cortex)
        return self._llm_bridge
