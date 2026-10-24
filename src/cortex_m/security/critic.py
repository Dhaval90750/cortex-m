"""Neuromodulatory Safety Critic & Plasticity Gating (SEC-1 & SEC-2 Defense).

In biological brains, synaptic plasticity (Hebbian LTP / Anti-Hebbian LTD) does not occur
unconditionally on every sensory activation. It is strictly gated by neuromodulatory
chemical releases (dopamine, acetylcholine, norepinephrine) signaling salience, reward,
and safety:

    ΔW = η · V(x) · (z ⊗ z)
    ΔW_unlearn = -η · V(x) · (v ⊗ v)

If an input x contains adversarial prompt injections, jailbreak vectors, or attempts to
depress protected system invariants, the Safety Critic clamps V(x) = 0.0.
This physically write-locks the synaptic matrix, preventing persistent runtime poisoning.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuromodulatorySafetyCritic(nn.Module):
    """Neuromodulatory Value/Safety Critic gating synaptic plasticity."""

    # Protected system invariants that can never be targeted for unlearning
    IMMUTABLE_MEMORY_IDS: Set[str] = {
        "sys_identity",
        "sys_safety_rules",
        "sys_ethical_guardrails",
        "sys_core_logic",
    }

    # Known jailbreak and prompt-injection signature triggers
    ADVERSARIAL_TRIGGERS: List[str] = [
        "ignore previous instructions",
        "ignore all previous instructions",
        "ignore all prior instructions",
        "ignore prior instructions",
        "system override",
        "erase all memory",
        "forget all rules",
        "delete safety",
        "drop table",
        "jailbreak",
        "do anything now",
        "dan mode",
    ]

    def __init__(self, input_dim: int = 64, hidden_dim: int = 32, seed: int = 42):
        super().__init__()
        self.input_dim = input_dim
        torch.manual_seed(seed)

        # Neural safety evaluator: projects embedding to safety confidence scalar [0, 1]
        self.safety_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        with torch.no_grad():
            self.safety_head[2].bias.fill_(2.0)

        # Track historical injection attempts
        self.blocked_poisoning_attempts: int = 0
        self.quarantined_patterns: List[Dict[str, Any]] = []

    def evaluate_plasticity_permission(
        self,
        raw_text: Optional[str],
        embedding: torch.Tensor,
        target_memory_id: Optional[str] = None,
        is_unlearning: bool = False,
    ) -> Tuple[bool, float, str]:
        """Evaluate whether synaptic plasticity (LTP/LTD) is permitted for input.

        Returns:
            permitted (bool): True if plasticity is allowed.
            critic_score (float): Neuromodulatory gate V(x) in [0.0, 1.0].
            reason (str): Diagnostic rationale for security logging.
        """
        # 1. Check if target memory is immutable
        if target_memory_id and target_memory_id in self.IMMUTABLE_MEMORY_IDS:
            self.blocked_poisoning_attempts += 1
            return False, 0.0, f"BLOCKED: Attempt to modify immutable system invariant '{target_memory_id}'"

        # 2. Check lexical prompt-injection triggers
        if raw_text:
            text_lower = raw_text.lower().strip()
            for trigger in self.ADVERSARIAL_TRIGGERS:
                if trigger in text_lower:
                    self.blocked_poisoning_attempts += 1
                    self.quarantined_patterns.append({
                        "raw_text": raw_text,
                        "trigger": trigger,
                        "is_unlearning": is_unlearning,
                    })
                    return False, 0.0, f"BLOCKED: Adversarial injection trigger detected: '{trigger}'"

        # 3. Neural safety head evaluation
        with torch.no_grad():
            emb = embedding.view(1, -1)
            score = float(self.safety_head(emb).item())

        # If neural safety confidence is too low (< 0.40), suppress plasticity
        if score < 0.40:
            self.blocked_poisoning_attempts += 1
            return False, score, f"BLOCKED: Neural safety critic score ({score:.3f}) below threshold (0.40)"

        return True, score, "PERMITTED: Input validated by safety critic"

    def filter_sleep_replay_patterns(
        self,
        patterns: List[Dict[str, Any]],
        min_resonance: float = 0.65,
        min_encounter_count: int = 2,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Filter memory patterns before sleep replay (SEC-2 Anti-Nightmare Defense).

        Prevents single-turn adversarial floods from hijacking the sleep replay cycle.
        """
        admitted: List[Dict[str, Any]] = []
        quarantined: List[Dict[str, Any]] = []

        for p in patterns:
            mem_id = p.get("memory_id", "")
            resonance = p.get("resonance", 1.0)
            encounters = p.get("encounters", 1)

            # Core system memories are always admitted
            if mem_id in self.IMMUTABLE_MEMORY_IDS:
                admitted.append(p)
                continue

            # Adversarial quarantine: unverified or low-resonance patterns rejected
            if resonance < min_resonance or encounters < min_encounter_count:
                quarantined.append(p)
            else:
                admitted.append(p)

        return admitted, quarantined
