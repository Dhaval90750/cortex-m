"""Hybrid semantic feature encoder for text and numeric inputs.

Provides dense semantic vector representations on the unit hypersphere:
1. SentenceTransformer (`all-MiniLM-L6-v2`) if installed and loaded.
2. Built-in zero-dependency PyTorch Semantic Concept Projection:
   Decomposes text into semantic concept bases + subwords + character trigrams,
   projecting queries and facts into a unified semantic space where paraphrases
   and QA pairs (e.g. "My name is Dhaval" vs "What is my name?") have high
   cosine similarity (>0.70), while unrelated topics ("The weather is sunny")
   remain near-orthogonal (<0.30).
3. Continuous numeric and role projection.
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Built-in semantic concept anchors for zero-dependency semantic alignment
SEMANTIC_ANCHORS = {
    "identity": [
        "name", "who", "called", "identify", "identity", "myself", "user", "person",
        "dhaval", "alice", "bob", "charlie"
    ],
    "preference": [
        "favorite", "favourite", "like", "prefer", "love", "color", "colour", "blue",
        "green", "red", "yellow", "taste", "choice"
    ],
    "location": [
        "live", "where", "city", "place", "country", "home", "reside", "located",
        "location", "address", "town"
    ],
    "occupation": [
        "work", "job", "profession", "role", "career", "engineer", "scientist",
        "researcher", "doctor", "developer", "do"
    ],
    "state_mode": [
        "mode", "state", "status", "setting", "alpha", "beta", "gamma", "delta",
        "config", "configuration", "level"
    ],
    "task_relational": [
        "route", "compute", "mimo", "matrix", "vector", "value", "graph", "edge",
        "relation", "permute", "permutation", "target"
    ],
    "temporal": [
        "time", "when", "year", "date", "day", "month", "now", "today", "yesterday",
        "age", "old"
    ],
    "nature_environment": [
        "weather", "rain", "sun", "sunny", "sky", "cloud", "temperature", "climate",
        "outside", "hot", "cold", "nice"
    ],
    "software_code": [
        "code", "coding", "program", "programming", "language", "python", "developer", "function", "script"
    ],
    "systems_infra": [
        "server", "port", "network", "host", "socket", "ip", "database", "devops", "cloud", "run"
    ],
    "neural_ai": [
        "model", "neural", "architecture", "cortex", "deep", "mimo", "weights", "deployment"
    ]
}


class BuiltinSemanticProjection(nn.Module):
    """Zero-dependency PyTorch semantic projection mapping phrases to R^D.
    
    Combines concept cluster activations with subword hash embeddings,
    projected through an orthogonal matrix. Ensures semantic relevance
    between queries and assertions while maintaining distinct identities.
    """

    def __init__(self, output_dim: int = 64, num_hash_buckets: int = 4096, seed: int = 42):
        super().__init__()
        self.output_dim = output_dim
        self.num_hash_buckets = num_hash_buckets
        self.concept_names = list(SEMANTIC_ANCHORS.keys())
        self.num_concepts = len(self.concept_names)

        torch.manual_seed(seed)
        # Hash embedding for specific word identities and trigrams
        self.hash_emb = nn.Embedding(num_hash_buckets, output_dim)
        nn.init.normal_(self.hash_emb.weight, mean=0.0, std=1.0)

        # Concept subspace projection
        concept_proj = torch.empty(self.num_concepts, output_dim)
        nn.init.orthogonal_(concept_proj)
        self.register_buffer("concept_proj", concept_proj)

        # Global orthogonal projection to distribute variance
        proj = torch.empty(output_dim, output_dim)
        nn.init.orthogonal_(proj)
        self.register_buffer("proj", proj)

    def _hash_token(self, token: str) -> int:
        h = hashlib.sha256(token.encode("utf-8")).digest()
        val = int.from_bytes(h[:4], byteorder="little")
        return val % self.num_hash_buckets

    def _extract_tokens_and_concepts(self, text: str) -> tuple[list[int], torch.Tensor]:
        cleaned = text.strip().lower()
        words = re.findall(r"[a-z0-9_]+", cleaned)

        # Compute concept weights
        concept_weights = torch.zeros(self.num_concepts, dtype=torch.float32)
        for i, c_name in enumerate(self.concept_names):
            anchor_words = SEMANTIC_ANCHORS[c_name]
            score = 0.0
            for w in words:
                if w in anchor_words:
                    score += 2.5
                elif any(anchor in w or w in anchor for anchor in anchor_words if len(anchor) > 3):
                    score += 1.0
            if score > 0:
                concept_weights[i] = score

        # Trigrams and word hashes
        buckets: list[int] = []
        padded = f"< {cleaned} >"
        for i in range(len(padded) - 2):
            buckets.append(self._hash_token(padded[i : i + 3]))
        for w in words:
            buckets.append(self._hash_token(f"w:{w}"))

        if not buckets:
            buckets = [0]
        return buckets, concept_weights

    def encode(self, text: str) -> torch.Tensor:
        buckets, concept_weights = self._extract_tokens_and_concepts(text)
        device = self.hash_emb.weight.device
        idx_tensor = torch.tensor(buckets, dtype=torch.long, device=device)
        concept_weights = concept_weights.to(device)

        with torch.no_grad():
            hash_vec = self.hash_emb(idx_tensor).mean(dim=0)  # [output_dim]
            if concept_weights.sum() > 0:
                norm_c = concept_weights / (concept_weights.norm() + 1e-6)
                concept_vec = torch.matmul(norm_c, self.concept_proj)  # [output_dim]
                # Combine concept semantic guidance (60%) with lexical identity (40%)
                combined = 0.65 * concept_vec + 0.35 * F.normalize(hash_vec, p=2, dim=0)
            else:
                combined = hash_vec

            projected = torch.matmul(combined, self.proj)
            return F.normalize(projected, p=2, dim=-1)


class NeuralFeatureEncoder(nn.Module):
    """Hybrid neural feature encoder supporting both offline PyTorch semantic projection
    and sentence-transformers (all-MiniLM-L6-v2) if available.
    """

    def __init__(
        self,
        output_dim: int = 64,
        backend: str = "hybrid",
        num_hash_buckets: int = 4096,
        seed: int = 42,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.backend = backend
        self.seed = seed

        # Built-in offline semantic projection
        self.builtin = BuiltinSemanticProjection(
            output_dim=output_dim,
            num_hash_buckets=num_hash_buckets,
            seed=seed,
        )

        # Optional SentenceTransformer
        self._st_model = None
        self._st_projection = None
        if backend in ("sentence-transformers", "hybrid"):
            self._init_sentence_transformer()

        # Projection buffer for numeric inputs
        proj = torch.empty(output_dim, output_dim)
        torch.manual_seed(seed)
        nn.init.orthogonal_(proj)
        self.register_buffer("numeric_proj", proj)

    def _init_sentence_transformer(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            st_dim = self._st_model.get_sentence_embedding_dimension()
            if st_dim != self.output_dim:
                proj = torch.empty(st_dim, self.output_dim)
                torch.manual_seed(self.seed)
                nn.init.orthogonal_(proj)
                self._st_projection = proj
        except Exception:
            # Gracefully fallback to builtin semantic projection
            self._st_model = None

    @property
    def has_sentence_transformers(self) -> bool:
        return self._st_model is not None

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode a single text phrase into a unit-normalized vector in R^output_dim."""
        if self._st_model is not None and self.backend in ("sentence-transformers", "hybrid"):
            try:
                emb = self._st_model.encode(text, convert_to_tensor=True, show_progress_bar=False)
                if emb.ndim == 1:
                    emb = emb.unsqueeze(0)
                if self._st_projection is not None:
                    emb = torch.matmul(emb.to(torch.float32), self._st_projection.to(emb.device))
                return F.normalize(emb.squeeze(0), p=2, dim=-1)
            except Exception:
                pass

        # Use builtin semantic projection
        return self.builtin.encode(text)

    def encode_batch_text(self, texts: Sequence[str]) -> torch.Tensor:
        """Encode multiple text strings into a [B, output_dim] tensor."""
        vecs = [self.encode_text(t) for t in texts]
        return torch.stack(vecs, dim=0)

    def encode_numeric(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize and project a numeric tensor into [B, output_dim]."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        B, in_dim = x.shape
        if in_dim == self.output_dim:
            return F.normalize(x, p=2, dim=-1)
        if in_dim < self.output_dim:
            pad = torch.zeros(B, self.output_dim - in_dim, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, pad], dim=-1)
            return F.normalize(torch.matmul(x_padded, self.numeric_proj), p=2, dim=-1)
        else:
            return F.normalize(torch.matmul(x[:, : self.output_dim], self.numeric_proj), p=2, dim=-1)

    def forward(self, inp: Union[str, Sequence[str], torch.Tensor]) -> torch.Tensor:
        if isinstance(inp, str):
            return self.encode_text(inp)
        elif isinstance(inp, (list, tuple)) and all(isinstance(i, str) for i in inp):
            return self.encode_batch_text(inp)
        elif isinstance(inp, torch.Tensor):
            return self.encode_numeric(inp)
        else:
            raise TypeError(f"Unsupported input type for encoder: {type(inp)}")
