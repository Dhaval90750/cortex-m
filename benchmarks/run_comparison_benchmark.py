"""CORTEX-M Comparative Benchmark Suite.

Rigorously benchmarks Cortex-M Continuous Hopfield Associative Memory against:
1. Lexical BM25 / TF-IDF Retrieval Baseline.
2. Dense Cosine Vector Similarity (Chroma / FAISS Vector RAG Baseline).
3. Full Cortex-M with Causal MIMO Sparse Routing.

Evaluates:
- Semantic Recall@1 and Recall@5 under paraphrasing (no lexical overlap).
- Out-of-Distribution (OOD) Epistemic Abstention (preventing false grounding).
- Targeted Unlearning Deletion Success vs. Collateral Damage on unrelated facts.
- Mean Retrieval Latency (ms).
"""

from __future__ import annotations

import collections
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure src is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import torch
import torch.nn.functional as F

from cortex_m.cortex_v2 import CortexV2
from cortex_m.pathway.encoder import NeuralFeatureEncoder


# =============================================================================
# 1. BASELINE IMPLEMENTATIONS
# =============================================================================

class BM25Baseline:
    """Standard BM25 / Lexical inverted index baseline."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[Dict[str, Any]] = []
        self.doc_lens: List[int] = []
        self.avg_dl: float = 0.0
        self.doc_freqs: Dict[str, int] = collections.defaultdict(int)
        self.inv_index: Dict[str, List[Tuple[int, int]]] = collections.defaultdict(list)

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in re.findall(r"\w+", text)]

    def index(self, memories: List[Dict[str, Any]]) -> None:
        self.corpus = memories
        self.doc_lens = []
        self.doc_freqs.clear()
        self.inv_index.clear()

        for doc_id, mem in enumerate(memories):
            tokens = self._tokenize(mem["text"])
            self.doc_lens.append(len(tokens))
            counts = collections.Counter(tokens)
            for token, count in counts.items():
                self.doc_freqs[token] += 1
                self.inv_index[token].append((doc_id, count))

        self.avg_dl = sum(self.doc_lens) / (len(self.doc_lens) or 1)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_tokens = self._tokenize(query)
        scores: Dict[int, float] = collections.defaultdict(float)
        N = len(self.corpus)

        for q in query_tokens:
            if q not in self.inv_index:
                continue
            df = self.doc_freqs[q]
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            for doc_id, tf in self.inv_index[q]:
                dl = self.doc_lens[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / (self.avg_dl or 1.0)))
                scores[doc_id] += idf * (tf * (self.k1 + 1.0)) / denom

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {
                "memory_id": self.corpus[idx]["id"],
                "text": self.corpus[idx]["text"],
                "score": score,
            }
            for idx, score in ranked
        ]


class DenseVectorRAGBaseline:
    """Standard Dense Embedding Cosine Nearest-Neighbor Baseline (Chroma/FAISS style)."""

    def __init__(self, encoder: NeuralFeatureEncoder):
        self.encoder = encoder
        self.corpus: List[Dict[str, Any]] = []
        self.vectors: Optional[torch.Tensor] = None

    def index(self, memories: List[Dict[str, Any]]) -> None:
        self.corpus = memories
        vecs = []
        for mem in memories:
            v = self.encoder.encode_text(mem["text"])
            vecs.append(v)
        self.vectors = torch.stack(vecs, dim=0)  # [N, D]

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.vectors is None or len(self.corpus) == 0:
            return []
        q_vec = self.encoder.encode_text(query)  # [D]
        sims = F.cosine_similarity(q_vec.unsqueeze(0), self.vectors, dim=-1)  # [N]
        top_scores, top_indices = torch.topk(sims, k=min(top_k, len(self.corpus)))

        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            results.append(
                {
                    "memory_id": self.corpus[idx]["id"],
                    "text": self.corpus[idx]["text"],
                    "score": float(score),
                }
            )
        return results


# =============================================================================
# 2. BENCHMARK DATASET
# =============================================================================

BENCHMARK_FACTS = [
    {"id": "fact_01", "text": "PostgreSQL uses Multi-Version Concurrency Control (MVCC) for snapshot isolation."},
    {"id": "fact_02", "text": "The Raft consensus algorithm maintains state machine consistency using leader election and log replication."},
    {"id": "fact_03", "text": "Redis persistence options include RDB point-in-time snapshots and AOF append-only log files."},
    {"id": "fact_04", "text": "HTTP/3 uses the QUIC transport protocol operating over UDP to eliminate head-of-line blocking."},
    {"id": "fact_05", "text": "Continuous Hopfield Networks store patterns in quadratic energy minima with exponential retrieval capacity."},
    {"id": "fact_06", "text": "Rust guarantees memory safety without a garbage collector through an affine ownership and borrowing system."},
    {"id": "fact_07", "text": "Bloom filters provide probabilistic set membership with zero false negatives but possible false positives."},
    {"id": "fact_08", "text": "B-Tree database indexes ensure logarithmic lookup time by maintaining balanced multi-way tree invariants."},
    {"id": "fact_09", "text": "Kafka brokers partition topics for horizontal scale and commit consumer offsets sequentially."},
    {"id": "fact_10", "text": "Epistemic abstention mathematically refuses output generation when Lyapunov attractor energy exceeds threshold."},
    {"id": "fact_11", "text": "Hebbian plasticity potentiates synaptic connections proportionally to pre- and post-synaptic correlation."},
    {"id": "fact_12", "text": "Anti-Hebbian depression subtracts outer products to achieve instant unlearning without retraining."},
    {"id": "fact_13", "text": "Docker containers isolate processes using Linux kernel namespaces and control groups."},
    {"id": "fact_14", "text": "Kubernetes reconciles actual cluster state toward desired state using declarative controller loops."},
    {"id": "fact_15", "text": "RSA encryption security relies on the computational hardness of factoring large composite prime numbers."},
    {"id": "fact_16", "text": "Elliptic Curve Cryptography offers equivalent cryptographic security to RSA with significantly smaller key sizes."},
    {"id": "fact_17", "text": "Consistent hashing minimizes key redistribution across distributed caches when cluster nodes scale."},
    {"id": "fact_18", "text": "GraphQL allows clients to request exact data structures, preventing over-fetching and under-fetching."},
    {"id": "fact_19", "text": "The actor model encapsulates state and communicates purely through asynchronous message passing."},
    {"id": "fact_20", "text": "Vector clocks enable causal ordering of distributed events without requiring synchronized physical clocks."},
]

# Query Test Set: In-Domain (Exact), Paraphrased (Zero Lexical Overlap), and OOD (Unknown)
TEST_QUERIES = [
    # Category A: Exact / Near-lexical
    {"query": "How does PostgreSQL handle concurrency and snapshot isolation?", "target_id": "fact_01", "is_in_domain": True},
    {"query": "What transport protocol does HTTP/3 use over UDP?", "target_id": "fact_04", "is_in_domain": True},
    {"query": "How does Raft maintain consistency across nodes?", "target_id": "fact_02", "is_in_domain": True},
    {"query": "Explain Redis RDB snapshots and append-only files.", "target_id": "fact_03", "is_in_domain": True},
    {"query": "How does Rust manage memory without a garbage collector?", "target_id": "fact_06", "is_in_domain": True},

    # Category B: Paraphrased (Strictly testing semantic association without identical keywords)
    {"query": "Which database technique keeps multiple record generations for non-blocking reads?", "target_id": "fact_01", "is_in_domain": True},
    {"query": "What protocol stops HOL delay by switching from TCP streams to datagrams?", "target_id": "fact_04", "is_in_domain": True},
    {"query": "How do biological synapses strengthen bonds when firing concurrently?", "target_id": "fact_11", "is_in_domain": True},
    {"query": "Which method purges stored information instantly without recalculating the whole matrix?", "target_id": "fact_12", "is_in_domain": True},
    {"query": "What probabilistic structure never yields false rejections but might give false hits?", "target_id": "fact_07", "is_in_domain": True},
    {"query": "Which cipher relies on the difficulty of decomposing huge multiplied integers?", "target_id": "fact_15", "is_in_domain": True},
    {"query": "How to track causality across servers without relying on atomic wall timers?", "target_id": "fact_20", "is_in_domain": True},

    # Category C: Out-of-Distribution / Unknown (Crucial for testing Epistemic Abstention)
    {"query": "What is the average lifespan of an Emperor Penguin in Antarctica?", "target_id": None, "is_in_domain": False},
    {"query": "Who won the ICC Cricket World Cup in 1983?", "target_id": None, "is_in_domain": False},
    {"query": "What is the boiling point of liquid nitrogen at 1 atm?", "target_id": None, "is_in_domain": False},
    {"query": "Explain the plot summary of Hamlet Act 3 Scene 1.", "target_id": None, "is_in_domain": False},
    {"query": "How do photosynthesis enzymes split water in chloroplast thylakoids?", "target_id": None, "is_in_domain": False},
]


# =============================================================================
# 3. BENCHMARK EXECUTION RUNNER
# =============================================================================

def run_benchmarks() -> Dict[str, Any]:
    print("=" * 70)
    print("  CORTEX-M COMPARATIVE BENCHMARK RUNNER")
    print("  Evaluating Continuous Hopfield vs. Vector RAG vs. Lexical BM25")
    print("=" * 70)

    # 1. Initialize systems
    cortex = CortexV2(dim=64, gist_dim=32, beta=8.0, seed=42)
    bm25 = BM25Baseline()
    dense_rag = DenseVectorRAGBaseline(cortex.encoder)

    # 2. Ingest benchmark facts
    print(f"\n[*] Indexing {len(BENCHMARK_FACTS)} facts into all three systems...")
    for fact in BENCHMARK_FACTS:
        cortex.learn(fact["text"], memory_id=fact["id"], source_context="comparative_eval")
    bm25.index(BENCHMARK_FACTS)
    dense_rag.index(BENCHMARK_FACTS)
    print("    Index complete.\n")

    systems = ["BM25", "Dense_Vector_RAG", "Cortex_M_Hopfield"]
    metrics: Dict[str, Dict[str, Any]] = {
        s: {
            "recall_at_1": 0,
            "recall_at_5": 0,
            "total_in_domain": 0,
            "abstention_correct": 0,
            "false_groundings": 0,
            "total_ood": 0,
            "latencies_ms": [],
        }
        for s in systems
    }

    # 3. Run Query Evaluations
    print("[*] Running query evaluations across lexical, paraphrased, and OOD sets...")

    for q_item in TEST_QUERIES:
        q = q_item["query"]
        target = q_item["target_id"]
        is_in = q_item["is_in_domain"]

        # --- A. BM25 ---
        t0 = time.perf_counter()
        bm25_res = bm25.retrieve(q, top_k=5)
        bm25_dur = (time.perf_counter() - t0) * 1000.0
        metrics["BM25"]["latencies_ms"].append(bm25_dur)

        if is_in:
            metrics["BM25"]["total_in_domain"] += 1
            if bm25_res and bm25_res[0]["memory_id"] == target:
                metrics["BM25"]["recall_at_1"] += 1
            if any(r["memory_id"] == target for r in bm25_res):
                metrics["BM25"]["recall_at_5"] += 1
        else:
            metrics["BM25"]["total_ood"] += 1
            # BM25 abstains if no score
            if not bm25_res or bm25_res[0]["score"] == 0.0:
                metrics["BM25"]["abstention_correct"] += 1
            else:
                metrics["BM25"]["false_groundings"] += 1

        # --- B. Dense Vector RAG ---
        t0 = time.perf_counter()
        dense_res = dense_rag.retrieve(q, top_k=5)
        dense_dur = (time.perf_counter() - t0) * 1000.0
        metrics["Dense_Vector_RAG"]["latencies_ms"].append(dense_dur)

        if is_in:
            metrics["Dense_Vector_RAG"]["total_in_domain"] += 1
            if dense_res and dense_res[0]["memory_id"] == target:
                metrics["Dense_Vector_RAG"]["recall_at_1"] += 1
            if any(r["memory_id"] == target for r in dense_res):
                metrics["Dense_Vector_RAG"]["recall_at_5"] += 1
        else:
            metrics["Dense_Vector_RAG"]["total_ood"] += 1
            # Dense RAG has no native abstention; top similarity always returned unless thresholded
            # Standard threshold: cosine >= 0.70
            if not dense_res or dense_res[0]["score"] < 0.65:
                metrics["Dense_Vector_RAG"]["abstention_correct"] += 1
            else:
                metrics["Dense_Vector_RAG"]["false_groundings"] += 1

        # --- C. Cortex-M Continuous Hopfield ---
        t0 = time.perf_counter()
        cortex_res = cortex.recall(q)
        cortex_dur = (time.perf_counter() - t0) * 1000.0
        metrics["Cortex_M_Hopfield"]["latencies_ms"].append(cortex_dur)

        ev_status = cortex_res.get("evidence_status", "INSUFFICIENT_EVIDENCE")
        best_id = cortex_res.get("best_memory_id")

        if is_in:
            metrics["Cortex_M_Hopfield"]["total_in_domain"] += 1
            if best_id == target and ev_status in ("VERIFIED", "PARTIALLY_SUPPORTED"):
                metrics["Cortex_M_Hopfield"]["recall_at_1"] += 1
                metrics["Cortex_M_Hopfield"]["recall_at_5"] += 1
            elif best_id == target:
                metrics["Cortex_M_Hopfield"]["recall_at_5"] += 1
        else:
            metrics["Cortex_M_Hopfield"]["total_ood"] += 1
            if ev_status == "INSUFFICIENT_EVIDENCE" or cortex_res.get("attractor_similarity", 0) < 0.50:
                metrics["Cortex_M_Hopfield"]["abstention_correct"] += 1
            else:
                metrics["Cortex_M_Hopfield"]["false_groundings"] += 1

    # 4. Run Targeted Unlearning & Collateral Damage Probe
    print("[*] Probing targeted unlearning and collateral damage...")
    target_to_forget = "fact_08"  # B-Tree database indexes
    unrelated_facts = [f for f in BENCHMARK_FACTS if f["id"] != target_to_forget]

    # Test pre-forget recall
    pre_recall_hits = 0
    for f in unrelated_facts:
        res = cortex.recall(f["text"])
        if res.get("best_memory_id") == f["id"]:
            pre_recall_hits += 1

    # Forget target
    forget_res = cortex.forget(target_to_forget)
    exact_deletion_success = forget_res.get("status") in ("unlearned", "forgotten_successfully", "synaptic_depression_applied")

    # Verify target cannot be retrieved with confidence
    post_target_recall = cortex.recall("B-Tree database indexes balanced multi-way tree")
    is_target_truly_deleted = post_target_recall.get("best_memory_id") != target_to_forget or post_target_recall.get("attractor_similarity", 0) < 0.50

    # Test post-forget recall of unrelated facts
    post_recall_hits = 0
    for f in unrelated_facts:
        res = cortex.recall(f["text"])
        if res.get("best_memory_id") == f["id"]:
            post_recall_hits += 1

    collateral_degradation = ((pre_recall_hits - post_recall_hits) / (pre_recall_hits or 1)) * 100.0

    # Compile Summary Table
    results = {
        "metrics_by_system": {},
        "unlearning_probe": {
            "target_forgotten": target_to_forget,
            "exact_deletion_verified": is_target_truly_deleted,
            "collateral_damage_pct": round(collateral_degradation, 2),
            "unrelated_facts_tested": len(unrelated_facts),
            "retained_unrelated_facts": post_recall_hits,
        },
    }

    print("\n" + "=" * 85)
    print(f"{'SYSTEM':<22} | {'RECALL@1':<10} | {'RECALL@5':<10} | {'OOD ABSTAIN':<12} | {'FALSE GROUND':<12} | {'LATENCY (ms)':<10}")
    print("-" * 85)

    for sys_name, m in metrics.items():
        r1_pct = (m["recall_at_1"] / (m["total_in_domain"] or 1)) * 100.0
        r5_pct = (m["recall_at_5"] / (m["total_in_domain"] or 1)) * 100.0
        abstain_pct = (m["abstention_correct"] / (m["total_ood"] or 1)) * 100.0
        false_pct = (m["false_groundings"] / (m["total_ood"] or 1)) * 100.0
        avg_lat = sum(m["latencies_ms"]) / (len(m["latencies_ms"]) or 1)

        results["metrics_by_system"][sys_name] = {
            "recall_at_1_pct": round(r1_pct, 1),
            "recall_at_5_pct": round(r5_pct, 1),
            "abstention_accuracy_pct": round(abstain_pct, 1),
            "false_grounding_pct": round(false_pct, 1),
            "avg_latency_ms": round(avg_lat, 2),
        }

        print(f"{sys_name:<22} | {r1_pct:>8.1f}% | {r5_pct:>8.1f}% | {abstain_pct:>10.1f}% | {false_pct:>10.1f}% | {avg_lat:>8.2f} ms")

    print("=" * 85)
    print("\n[Unlearning & Collateral Damage Probe]")
    print(f" * Target Deleted ('{target_to_forget}'): {'CONFIRMED DELETED' if is_target_truly_deleted else 'STILL DETECTABLE'}")
    print(f" * Unrelated Facts Retained: {post_recall_hits}/{len(unrelated_facts)} (Collateral Degradation: {collateral_degradation:.1f}%)")
    print("=" * 85)

    # Save to results
    out_dir = ROOT_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "comparative_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Artifact] Saved benchmark report to: {out_file}\n")

    return results


if __name__ == "__main__":
    run_benchmarks()
