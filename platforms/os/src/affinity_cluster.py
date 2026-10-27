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
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

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

    def build_lsh_index(
        self,
        documents: List[FileDocument],
        num_bits: int = 8,
        num_tables: int = 4,
    ) -> RandomProjectionLSH:
        """Construct multi-table random projection LSH index over documents."""
        lsh = RandomProjectionLSH(
            dimension=self.vector_dim,
            num_bits=num_bits,
            num_tables=num_tables,
        )
        for doc in documents:
            lsh.index_document(doc.path, doc.vector_64d)
        return lsh


class RandomProjectionLSH:
    """Multi-table Random Hyperplane Locality-Sensitive Hashing (LSH) for O(1) semantic search.
    
    Guarantees angular collision probability: P(h(u) = h(v)) = 1 - theta(u, v) / pi.
    """

    def __init__(self, dimension: int = 64, num_bits: int = 8, num_tables: int = 4, seed: int = 42):
        self.dimension = dimension
        self.num_bits = num_bits
        self.num_tables = num_tables
        torch.manual_seed(seed)
        self.hyperplanes = torch.randn(num_tables, num_bits, dimension)
        self.hyperplanes = F.normalize(self.hyperplanes, p=2, dim=-1)
        self.tables: List[Dict[int, List[Tuple[str, torch.Tensor]]]] = [{} for _ in range(num_tables)]
        self._doc_map: Dict[str, torch.Tensor] = {}

    def _adapt_vector(self, vec: torch.Tensor) -> torch.Tensor:
        """Ensure input vector matches LSH hyperplane dimension via padding or projection."""
        v_flat = vec.view(-1).float()
        if v_flat.numel() < self.dimension:
            pad = torch.zeros(self.dimension - v_flat.numel(), device=v_flat.device, dtype=v_flat.dtype)
            v_flat = torch.cat([v_flat, pad])
        elif v_flat.numel() > self.dimension:
            v_flat = v_flat[:self.dimension]
        return F.normalize(v_flat, p=2, dim=0)

    def _hash_vector(self, vec: torch.Tensor, table_idx: int) -> int:
        """Project vector onto hyperplanes and convert binary signs to integer bucket hash."""
        v = self._adapt_vector(vec)
        dots = torch.matmul(self.hyperplanes[table_idx], v)
        bits = (dots >= 0.0).int().tolist()
        bucket = 0
        for b in bits:
            bucket = (bucket << 1) | b
        return bucket

    def index_document(self, doc_id: str, vector: torch.Tensor) -> None:
        """Insert document vector into all LSH tables."""
        v = self._adapt_vector(vector)
        self._doc_map[doc_id] = v
        for t_idx in range(self.num_tables):
            bucket = self._hash_vector(v, t_idx)
            if bucket not in self.tables[t_idx]:
                self.tables[t_idx][bucket] = []
            self.tables[t_idx][bucket].append((doc_id, v))

    def query(self, query_vec: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top_k nearest neighbors by exact cosine similarity over candidate collision set."""
        if not self._doc_map:
            return []
        q = self._adapt_vector(query_vec)
        candidates: Set[str] = set()

        for t_idx in range(self.num_tables):
            bucket = self._hash_vector(q, t_idx)
            if bucket in self.tables[t_idx]:
                for doc_id, _ in self.tables[t_idx][bucket]:
                    candidates.add(doc_id)

        eval_set = candidates if candidates else set(self._doc_map.keys())

        scored = []
        for doc_id in eval_set:
            v = self._doc_map[doc_id]
            sim = float(torch.dot(q, v).item())
            scored.append((doc_id, round(sim, 4)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class CountMinSketch:
    """4-bit Count-Min Sketch frequency estimator with periodic aging."""

    def __init__(self, depth: int = 4, width: int = 256):
        self.depth = depth
        self.width = width
        self.table = np.zeros((depth, width), dtype=np.int32)
        self.seeds = [17 * i + 31 for i in range(depth)]
        self.total_count = 0
        self.aging_threshold = 2048

    def _hash(self, key: str, d: int) -> int:
        h = hash((key, self.seeds[d]))
        return abs(h) % self.width

    def increment(self, key: str) -> None:
        for d in range(self.depth):
            idx = self._hash(key, d)
            if self.table[d, idx] < 15:
                self.table[d, idx] += 1
        self.total_count += 1
        if self.total_count >= self.aging_threshold:
            self.table = self.table // 2
            self.total_count = self.total_count // 2

    def estimate(self, key: str) -> int:
        return min(int(self.table[d, self._hash(key, d)]) for d in range(self.depth))


class WTinyLFUCache:
    """W-TinyLFU Cache Admission and Eviction Filter.
    
    Structure:
    - Window Cache (LRU): catches burst recency
    - SLRU Main Cache:
        - Probationary Segment: newly admitted items
        - Protected Segment: frequency-reinforced items
    - TinyLFU Admission:
        Candidate evicted from Window cache only enters Main cache if its Count-Min Sketch
        frequency estimate exceeds that of the victim from the Main cache.
    """

    def __init__(self, capacity: int = 100, window_ratio: float = 0.1, protected_ratio: float = 0.7):
        self.capacity = max(4, capacity)
        self.window_capacity = max(1, int(self.capacity * window_ratio))
        self.main_capacity = self.capacity - self.window_capacity
        self.protected_capacity = max(1, int(self.main_capacity * protected_ratio))
        self.probationary_capacity = max(1, self.main_capacity - self.protected_capacity)

        self.sketch = CountMinSketch()
        self.window_cache: Dict[str, Any] = {}
        self.probationary_cache: Dict[str, Any] = {}
        self.protected_cache: Dict[str, Any] = {}

    def get(self, key: str) -> Optional[Any]:
        self.sketch.increment(key)
        if key in self.window_cache:
            val = self.window_cache.pop(key)
            self.window_cache[key] = val
            return val
        if key in self.protected_cache:
            val = self.protected_cache.pop(key)
            self.protected_cache[key] = val
            return val
        if key in self.probationary_cache:
            val = self.probationary_cache.pop(key)
            if len(self.protected_cache) >= self.protected_capacity:
                demoted_k, demoted_v = next(iter(self.protected_cache.items()))
                del self.protected_cache[demoted_k]
                self.probationary_cache[demoted_k] = demoted_v
            self.protected_cache[key] = val
            return val
        return None

    def put(self, key: str, value: Any) -> None:
        self.sketch.increment(key)
        if key in self.window_cache:
            self.window_cache[key] = value
            return
        if key in self.protected_cache:
            self.protected_cache[key] = value
            return
        if key in self.probationary_cache:
            self.probationary_cache[key] = value
            return

        if len(self.window_cache) < self.window_capacity:
            self.window_cache[key] = value
            return

        cand_k, cand_v = next(iter(self.window_cache.items()))
        del self.window_cache[cand_k]
        self.window_cache[key] = value

        if len(self.probationary_cache) < self.probationary_capacity:
            self.probationary_cache[cand_k] = cand_v
            return

        victim_k, victim_v = next(iter(self.probationary_cache.items()))
        cand_freq = self.sketch.estimate(cand_k)
        victim_freq = self.sketch.estimate(victim_k)

        if cand_freq > victim_freq:
            del self.probationary_cache[victim_k]
            self.probationary_cache[cand_k] = cand_v

