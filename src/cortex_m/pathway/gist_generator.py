"""Pre-Verbal Gist Natural Response Realizer.

In human communication:
1. Gist is formed first (pre-verbal semantic intent).
2. Episodic memory attractor completes the pattern.
3. Language formulation translates internal gist + attractor into verbal response.
4. Confidence is modulated by pathway resonance (did recall activate the original pathway?).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .gist_bottleneck import GistBottleneck


class GistRealizer:
    """Realizes verbal expressions from pre-verbal gist vectors and synaptic attractors."""

    def __init__(self, high_confidence_threshold: float = 0.65):
        self.high_confidence_threshold = high_confidence_threshold

    def formulate_response(
        self,
        gist: torch.Tensor,
        recalled_metadata: Optional[Dict[str, Any]],
        resonance: Optional[Dict[str, Any]] = None,
        source_attribution: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Formulate a natural verbal response driven by pre-verbal gist and pathway resonance."""
        meta = recalled_metadata or {}
        raw_text = meta.get("raw_text")
        memory_id = meta.get("memory_id", "unknown")
        source = meta.get("source_context")

        res_score = float(resonance.get("resonance_score", 0.0)) if resonance else 0.0
        is_resonant = bool(res_score >= self.high_confidence_threshold)

        if not raw_text:
            text = f"Neural pattern completed into synaptic attractor basin '{memory_id}' (resonance: {res_score:.2f})."
        elif is_resonant:
            if source:
                text = f"{raw_text} (remembered from {source}, pathway resonance {res_score:.2f})."
            else:
                text = f"{raw_text} (pathway resonance {res_score:.2f})."
        else:
            text = f"Weak neural pathway recall (resonance {res_score:.2f}): possibly '{raw_text}'."

        return {
            "verbal_response": text,
            "raw_recalled_content": raw_text,
            "source_context": source,
            "pathway_resonance": res_score,
            "is_high_confidence": is_resonant,
            "pre_verbal_gist_dim": gist.shape[0] if gist is not None else 0,
        }
