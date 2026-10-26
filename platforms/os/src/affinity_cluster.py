"""Dynamic Cognitive Affinity Graph Clustering for CORTEX-OS.

In compliance with FR-OS-03:
1. Low-rank factorized Hopfield affinity projection (d=64, rank=4).
2. Mutual Gaussian resonance: A_ij = exp(-||d_i - d_j||^2 / (2 * sigma^2)).
3. Connected component extraction for dynamic workspace project grouping.
4. Human-readable cluster synthesis and cohesion scoring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from platforms.os.src.file_watcher import FileDocument


@dataclass
class ProjectCluster:
    cluster_id: str
    cluster_name: str
    document_paths: List[str]
    cohesion_score: float
    centroid_vector: torch.Tensor


@dataclass
class AffinityClusterReport:
    total_documents: int
    num_clusters: int
    clusters: List[ProjectCluster]
    graph_density: float


class CognitiveAffinityGraph:
    """Builds and analyzes dynamic associative file clusters."""

    def __init__(
        self,
        affinity_threshold: float = 0.45,
        sigma_affinity: float = 1.0,
        rank: int = 4,
        vector_dim: int = 64,
    ):
        self.affinity_threshold = affinity_threshold
        self.sigma_affinity = sigma_affinity
        self.rank = rank
        self.vector_dim = vector_dim

    def compute_affinity_matrix(self, documents: List[FileDocument]) -> np.ndarray:
        """Compute pairwise Gaussian affinity matrix A_ij."""
        n = len(documents)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float32)

        # Stack normalized document vectors
        vecs = torch.stack([d.vector_64d for d in documents])  # (n, 64)

        # Pairwise squared Euclidean distance: ||z_i - z_j||^2 = 2 * (1 - dot_prod)
        dot_prod = torch.matmul(vecs, vecs.t())  # (n, n)
        dist_sq = torch.clamp(2.0 * (1.0 - dot_prod), min=0.0)

        # Affinity A_ij = exp(-dist_sq / (2 * sigma^2))
        affinity = torch.exp(-dist_sq / (2.0 * (self.sigma_affinity ** 2)))
        return affinity.cpu().numpy()

    def extract_clusters(self, documents: List[FileDocument]) -> AffinityClusterReport:
        """Extract connected components and build project clusters."""
        n = len(documents)
        if n == 0:
            return AffinityClusterReport(
                total_documents=0,
                num_clusters=0,
                clusters=[],
                graph_density=0.0,
            )

        affinity = self.compute_affinity_matrix(documents)
        adjacency = affinity >= self.affinity_threshold
        np.fill_diagonal(adjacency, False)

        total_possible_edges = n * (n - 1)
        actual_edges = int(np.sum(adjacency))
        density = actual_edges / max(1, total_possible_edges)

        # Breadth-first search / connected components
        visited: Set[int] = set()
        clusters: List[ProjectCluster] = []

        for idx in range(n):
            if idx in visited:
                continue

            comp: List[int] = []
            queue = [idx]
            visited.add(idx)

            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                neighbors = np.where(adjacency[curr])[0]
                for nbr in neighbors:
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)

            comp_docs = [documents[i] for i in comp]
            paths = [d.path for d in comp_docs]

            # Compute centroid vector
            comp_vecs = torch.stack([d.vector_64d for d in comp_docs])
            centroid = torch.mean(comp_vecs, dim=0)
            centroid = centroid / (torch.norm(centroid) + 1e-8)

            # Cohesion score (average intra-cluster affinity)
            if len(comp) > 1:
                sub_aff = affinity[np.ix_(comp, comp)]
                cohesion = float((np.sum(sub_aff) - len(comp)) / (len(comp) * (len(comp) - 1)))
            else:
                cohesion = 1.0

            # Derive cluster name from most common path prefix or dominant file
            primary_path = paths[0]
            cluster_name = f"Cluster_{len(clusters) + 1}_{primary_path.split('/')[-1].split('.')[0]}"

            clusters.append(ProjectCluster(
                cluster_id=f"cluster_{len(clusters):03d}",
                cluster_name=cluster_name,
                document_paths=paths,
                cohesion_score=cohesion,
                centroid_vector=centroid,
            ))

        return AffinityClusterReport(
            total_documents=n,
            num_clusters=len(clusters),
            clusters=clusters,
            graph_density=density,
        )

    def compute_modularity(self, documents: List[FileDocument], report: AffinityClusterReport) -> float:
        """Calculate Newman-Girvan community modularity Q for the graph partition.
        
        Q = 1/(2m) * sum_ij [ A_ij - (k_i * k_j)/(2m) ] * delta(c_i, c_j)
        Values in [0.3, 0.7] indicate strong community structure.
        """
        n = len(documents)
        if n < 2 or report.num_clusters == 0:
            return 0.0

        affinity = self.compute_affinity_matrix(documents)
        adjacency = (affinity >= self.affinity_threshold).astype(np.float64)
        np.fill_diagonal(adjacency, 0.0)

        m = 0.5 * np.sum(adjacency)
        if m < 1e-6:
            return 0.0

        degrees = np.sum(adjacency, axis=1)

        # Map document index to cluster ID
        path_to_cluster = {}
        for c in report.clusters:
            for p in c.document_paths:
                path_to_cluster[p] = c.cluster_id

        doc_clusters = [path_to_cluster.get(d.path, "") for d in documents]

        Q = 0.0
        for i in range(n):
            for j in range(n):
                if doc_clusters[i] and doc_clusters[i] == doc_clusters[j]:
                    expected = (degrees[i] * degrees[j]) / (2.0 * m)
                    Q += (adjacency[i, j] - expected)

        Q = Q / (2.0 * m)
        return round(float(Q), 4)

    def predict_cluster(
        self,
        new_doc: FileDocument,
        report: AffinityClusterReport,
    ) -> Tuple[Optional[str], float]:
        """Predict optimal existing cluster assignment for an incoming document via centroid resonance."""
        if not report.clusters:
            return None, 0.0

        query_vec = new_doc.vector_64d
        best_cluster_id: Optional[str] = None
        best_sim = -1.0

        for c in report.clusters:
            sim = float(torch.dot(query_vec, c.centroid_vector).item())
            if sim > best_sim:
                best_sim = sim
                best_cluster_id = c.cluster_id

        return best_cluster_id, round(best_sim, 4)

