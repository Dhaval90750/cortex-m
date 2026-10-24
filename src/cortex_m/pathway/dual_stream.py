"""Dual-Stream Cortical Architecture & Multi-Head Factored Gist (EXP-1 & EXP-2).

Resolves two core expressivity limitations:
1. Loss of Hyper-Specific Details:
   In biological cognitive neuroscience (Goodale & Milner Dual-Stream Hypothesis):
   - Ventral Stream ("What" - Semantic Gist): Traverses the information bottleneck
     (64 -> 32 -> 64) extracting high-level intent, mode, and routing targets.
   - Dorsal Stream ("How / Action" - High-Fidelity Syntactic Scratchpad): Bypasses the
     bottleneck, preserving literal token IDs, code syntax tokens, AST markers,
     and exact numeric values for code generation and mathematical calculations.
2. Semantic Hash Collisions in 32-D Gist:
   Replaces monolithic 32-D gist with a 4-Head Factored Gist (4 x 32 = 128-dim):
   - g_action: Task intent (query, execute, compute, refute, summarize)
   - g_entity: Primary conceptual target (finance, rust_code, devops, bio_neural)
   - g_modality: Realization mode (code, terse, formal_prose, numeric)
   - g_constraint: Active operational constraints (strict_json, sub_50_words, causal_only)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadFactoredGist(nn.Module):
    """4-Head Factored Gist space (128-dim partitioned intent)."""

    HEAD_NAMES = ["action", "entity", "modality", "constraint"]

    def __init__(self, in_dim: int = 64, head_dim: int = 32, seed: int = 42):
        super().__init__()
        self.in_dim = in_dim
        self.head_dim = head_dim
        self.total_gist_dim = head_dim * len(self.HEAD_NAMES)  # 128-dim

        torch.manual_seed(seed)
        # 4 specialized projection heads
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(in_dim, head_dim),
                nn.Tanh(),
            )
            for name in self.HEAD_NAMES
        })

        # Multi-head expansion back to routing context
        self.context_reconstructor = nn.Sequential(
            nn.Linear(self.total_gist_dim, in_dim),
        )

    def extract_factored_gist(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract partitioned 4-head gist representations."""
        if x.ndim == 1:
            x_in = x.unsqueeze(0)
        else:
            x_in = x

        factored = {}
        for name, head in self.heads.items():
            out = head(x_in)
            factored[name] = F.normalize(out.squeeze(0) if x.ndim == 1 else out, p=2, dim=-1)
        return factored

    def concatenate_gist(self, factored: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate 4 heads into a unified normalized 128-dim vector."""
        tensors = [factored[name] for name in self.HEAD_NAMES]
        dim = -1
        unified = torch.cat(tensors, dim=dim)
        return F.normalize(unified, p=2, dim=dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns (unified_128d_gist, expanded_context, factored_dict)."""
        factored = self.extract_factored_gist(x)
        unified_gist = self.concatenate_gist(factored)
        if unified_gist.ndim == 1:
            u_in = unified_gist.unsqueeze(0)
            context = self.context_reconstructor(u_in).squeeze(0)
        else:
            context = self.context_reconstructor(unified_gist)
        return unified_gist, context, factored


class DorsalSyntacticBuffer:
    """High-bandwidth Dorsal stream preserving exact lexical/code tokens and numeric operands."""

    def __init__(self):
        self.raw_text_history: List[str] = []
        self.code_blocks: List[str] = []
        self.numeric_tokens: List[float] = []

    def ingest(self, text: str) -> Dict[str, Any]:
        """Parse and preserve verbatim literal structures."""
        self.raw_text_history.append(text)

        # Extract code blocks
        code_matches = re.findall(r"```(?:\w+)?\n?(.*?)```", text, re.DOTALL)
        for c in code_matches:
            self.code_blocks.append(c.strip())

        # Extract numeric tokens
        numbers = [float(n) for n in re.findall(r"[-+]?\d*\.?\d+", text) if n.strip()]
        self.numeric_tokens.extend(numbers)

        return {
            "has_code": len(code_matches) > 0,
            "code_count": len(code_matches),
            "numbers_detected": numbers,
            "verbatim_length": len(text),
        }

    def retrieve_exact_code(self) -> Optional[str]:
        return self.code_blocks[-1] if self.code_blocks else None

    def retrieve_exact_numbers(self) -> List[float]:
        return list(self.numeric_tokens)


class DualStreamProcessor(nn.Module):
    """Unifies Ventral semantic gist intent with Dorsal literal syntax."""

    def __init__(self, in_dim: int = 64, dim: Optional[int] = None, head_dim: int = 32, seed: int = 42):
        super().__init__()
        effective_dim = dim if dim is not None else in_dim
        self.ventral_gist = MultiHeadFactoredGist(in_dim=effective_dim, head_dim=head_dim, seed=seed)
        self.dorsal_buffer = DorsalSyntacticBuffer()

    def process(self, text: str, embedding: torch.Tensor) -> Dict[str, Any]:
        """Dual-stream forward execution."""
        # Dorsal stream preserves exact literals
        dorsal_meta = self.dorsal_buffer.ingest(text)

        # Ventral stream extracts multi-head factored intent
        unified_gist, context, factored = self.ventral_gist(embedding)

        return {
            "unified_gist": unified_gist,
            "unified_gist_dim": unified_gist.shape[-1],
            "context": context,
            "factored_heads": {k: [round(float(v), 3) for v in val.view(-1)[:4].tolist()] for k, val in factored.items()},
            "dorsal_meta": dorsal_meta,
            "verbatim_code": self.dorsal_buffer.retrieve_exact_code(),
            "verbatim_numbers": self.dorsal_buffer.retrieve_exact_numbers(),
        }
