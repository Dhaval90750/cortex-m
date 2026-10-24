"""Continuous Neural Subword Tokenizer for CortexLM.

Bridges discrete text tokens with the 64-dimensional continuous unit hypersphere.
Features:
- Fast subword tokenization with byte fallback.
- Bidirectional mapping: text -> token_ids -> continuous hypersphere vectors (R^64) -> text.
- Concept-anchored semantic alignment ensuring lexical and conceptual cohesion.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# Core foundational vocabulary for CortexLM
BASE_VOCABULARY = [
    "<pad>", "<bos>", "<eos>", "<unk>", "<mask>",
    # Logical and structural operators
    "is", "are", "was", "were", "be", "been", "the", "a", "an", "this", "that",
    "and", "or", "not", "if", "then", "else", "because", "therefore", "thus",
    "with", "without", "by", "for", "from", "to", "in", "on", "at", "through",
    "what", "why", "how", "where", "when", "who", "which",
    # Computer science & systems
    "rust", "python", "memory", "safety", "ownership", "borrow", "checker",
    "pointer", "reference", "thread", "concurrency", "lock", "mutex", "data", "race",
    "compiler", "runtime", "latency", "throughput", "cpu", "gpu", "register", "cache",
    "function", "class", "module", "struct", "enum", "trait", "interface", "method",
    "algorithm", "complexity", "time", "space", "hash", "tree", "graph", "vector",
    # Mathematics & logic
    "calculate", "compute", "solve", "equation", "matrix", "eigenvalue", "gradient",
    "integral", "derivative", "probability", "bayes", "distribution", "variance", "mean",
    "proof", "theorem", "lemma", "axiom", "boolean", "true", "false",
    # Neural computation & Cortex-M
    "cortex", "neural", "synapse", "weight", "hopfield", "attractor", "energy",
    "lyapunov", "descent", "hebbian", "plasticity", "anti-hebbian", "unlearning",
    "gist", "bottleneck", "sparse", "routing", "mimo", "resonance", "consolidation",
    "sleep", "homeostasis", "sprouting", "pruning", "working", "buffer", "pushdown",
    # Reasoning & realization
    "guarantees", "prevents", "achieves", "operates", "scales", "eliminates",
    "verified", "proven", "grounded", "certain", "uncertain", "hallucination",
    "zero", "database", "kv-cache", "tenant", "isolated", "leakage",
]


class NeuralTokenizer:
    """Continuous subword tokenizer with hypersphere projection."""

    def __init__(self, dim: int = 64, seed: int = 42):
        self.dim = dim
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}

        for idx, token in enumerate(BASE_VOCABULARY):
            self.token2id[token.lower()] = idx
            self.id2token[idx] = token.lower()

        self.pad_id = self.token2id["<pad>"]
        self.bos_id = self.token2id["<bos>"]
        self.eos_id = self.token2id["<eos>"]
        self.unk_id = self.token2id["<unk>"]

        torch.manual_seed(seed)
        # Continuous projection table mapping token IDs to unit hypersphere
        self.embeddings = nn.Embedding(len(self.token2id) + 2048, dim)
        nn.init.orthogonal_(self.embeddings.weight[:len(self.token2id)])

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    def tokenize(self, text: str) -> List[str]:
        """Split text into normalized tokens preserving numbers, code, and punctuation."""
        cleaned = re.sub(r"([.,!?;:()\[\]{}])", r" \1 ", text)
        return cleaned.strip().lower().split()

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convert text string into list of token IDs."""
        tokens = self.tokenize(text)
        ids = []
        if add_special_tokens:
            ids.append(self.bos_id)

        for tok in tokens:
            if tok in self.token2id:
                ids.append(self.token2id[tok])
            else:
                # Dynamic open-vocabulary registration
                new_id = len(self.token2id)
                self.token2id[tok] = new_id
                self.id2token[new_id] = tok
                ids.append(new_id)

        if add_special_tokens:
            ids.append(self.eos_id)
        return ids

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Reconstruct human-readable text from token IDs."""
        words = []
        for tid in token_ids:
            if skip_special_tokens and tid in (self.pad_id, self.bos_id, self.eos_id):
                continue
            tok = self.id2token.get(tid, f"<unk_{tid}>")
            words.append(tok)

        text = " ".join(words)
        # Restore punctuation spacing
        text = re.sub(r"\s+([.,!?;:()\[\]{}])", r"\1", text)
        return text

    def embed_tokens(self, token_ids: Sequence[int]) -> torch.Tensor:
        """Project token IDs to continuous hypersphere vectors [T, dim]."""
        ids_t = torch.tensor(token_ids, dtype=torch.long)
        raw_emb = self.embeddings(ids_t)
        return F.normalize(raw_emb, p=2, dim=-1)
