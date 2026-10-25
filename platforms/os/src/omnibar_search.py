"""Spotlight Associative Omnibar Search Engine for CORTEX-OS.

In compliance with FR-OS-02 & NFSR-OS-02:
1. Zero-latency conceptual search (< 1.0 ms evaluation for candidate pools).
2. Continuous Hopfield attractor memory query relaxation.
3. Top-k ranked results with path, snippet preview, and resonance affinity score.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from platforms.os.src.file_watcher import FileDocument, NonPrivilegedFileWatcher


@dataclass
class SearchResult:
    rank: int
    file_id: str
    file_path: str
    resonance_score: float
    snippet_preview: str


@dataclass
class OmnibarQueryResponse:
    query: str
    results: List[SearchResult]
    evaluation_latency_ms: float
    total_candidates: int


class SpotlightOmnibarEngine:
    """Spotlight conceptual search engine powered by continuous Hopfield dynamics."""

    def __init__(self, beta: float = 8.0, hopfield_iterations: int = 3):
        self.beta = beta
        self.hopfield_iterations = hopfield_iterations
        self.file_watcher = NonPrivilegedFileWatcher()
        # Warmup PyTorch BLAS kernel
        _dummy = torch.randn(2, 64)
        _ = torch.mv(_dummy, torch.randn(64))

    def query(
        self,
        query_text: str,
        candidates: List[FileDocument],
        top_k: int = 5,
    ) -> OmnibarQueryResponse:
        """Execute fast parallel dot-product and attractor relaxation search."""
        start_t = time.perf_counter()

        if not candidates:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return OmnibarQueryResponse(
                query=query_text,
                results=[],
                evaluation_latency_ms=elapsed_ms,
                total_candidates=0,
            )

        # 1. Encode query string into normalized 64-D vector
        q_vec = self.file_watcher.extract_gist_vector(query_text, "query://stdin")

        # 2. Stack candidate document vectors (N, 64)
        doc_matrix = torch.stack([d.vector_64d for d in candidates])  # Shape: (N, 64)

        # 3. Continuous Hopfield Attractor Relaxation
        # State vector s initialized to query
        s = q_vec.clone()

        with torch.no_grad():
            for _ in range(self.hopfield_iterations):
                # Projection: logits = beta * (D @ s)
                sims = torch.mv(doc_matrix, s)
                attn = F.softmax(self.beta * sims, dim=0)
                # Attractor pull: update s
                s = torch.mv(doc_matrix.t(), attn)
                s = s / (torch.norm(s) + 1e-8)

            # Final resonance scores
            resonance = torch.mv(doc_matrix, s).cpu().numpy()

        # 4. Rank candidates by descending score
        top_indices = torch.topk(
            torch.from_numpy(resonance),
            k=min(top_k, len(candidates)),
        ).indices.tolist()

        results: List[SearchResult] = []
        for rank, idx in enumerate(top_indices, start=1):
            doc = candidates[idx]
            results.append(SearchResult(
                rank=rank,
                file_id=doc.file_id,
                file_path=doc.path,
                resonance_score=float(resonance[idx]),
                snippet_preview=doc.content_preview,
            ))

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return OmnibarQueryResponse(
            query=query_text,
            results=results,
            evaluation_latency_ms=elapsed_ms,
            total_candidates=len(candidates),
        )
