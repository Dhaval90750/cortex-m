"""High-Throughput Financial Event Replay Engine for CORTEX-Fin.

Simulates and stress-tests payment transaction streams, benchmarking:
1. Replay throughput (transactions/second).
2. Latency percentiles (mean, p50, p95, p99 in microseconds).
3. Shadow-mode escalation rates and anomaly distributions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .anomaly_scorer import FinancialAnomalyScorer, FraudScoringReport
from .transaction_encoder import FinancialTransaction


@dataclass
class ReplayBatchResult:
    """Benchmark performance report across a replayed transaction stream."""
    total_transactions: int
    throughput_tx_per_sec: float
    total_duration_ms: float
    mean_latency_us: float
    p50_latency_us: float
    p95_latency_us: float
    p99_latency_us: float
    passed_count: int
    review_count: int
    escalated_count: int
    reports: List[FraudScoringReport] = field(default_factory=list)


class HighThroughputReplayEngine:
    """Executes high-frequency transaction event replay benchmarks."""

    def __init__(self, scorer: FinancialAnomalyScorer, encoder: Optional[Any] = None):
        self.scorer = scorer
        self.encoder = encoder

    def benchmark_stream(self, count: int = 100) -> Dict[str, Any]:
        """Generate synthetic stream of transactions and return latency & throughput metrics."""
        np.random.seed(42)
        transactions: List[FinancialTransaction] = []
        for i in range(count):
            amount = float(np.random.exponential(scale=150.0) + 5.0)
            dist = float(np.random.exponential(scale=10.0))
            is_anomaly = (i % 25 == 0)
            if is_anomaly:
                amount *= 12.0
                dist += 120.0
            tx = FinancialTransaction(
                tx_id=f"tx_bench_{i:04d}",
                account_id=f"acc_{1000 + (i % 20)}",
                amount_usd=round(amount, 2),
                mcc=6051 if is_anomaly else 5411,
                geo_distance_km=round(dist, 1),
                time_delta_seconds=float(np.random.randint(10, 7200)),
            )
            transactions.append(tx)

        res = self.replay_stream(transactions, capture_individual_reports=False)
        return {
            "total_transactions": res.total_transactions,
            "throughput_tps": res.throughput_tx_per_sec,
            "p50_latency_ms": round(res.p50_latency_us / 1000.0, 3),
            "p95_latency_ms": round(res.p95_latency_us / 1000.0, 3),
            "p99_latency_ms": round(res.p99_latency_us / 1000.0, 3),
            "passed_count": res.passed_count,
            "review_count": res.review_count,
            "escalated_count": res.escalated_count,
        }

    def replay_stream(
        self,
        transactions: Sequence[FinancialTransaction],
        capture_individual_reports: bool = True,
    ) -> ReplayBatchResult:
        """Stream a batch of transactions through the scoring engine and capture telemetry."""
        total = len(transactions)
        if total == 0:
            return ReplayBatchResult(
                total_transactions=0,
                throughput_tx_per_sec=0.0,
                total_duration_ms=0.0,
                mean_latency_us=0.0,
                p50_latency_us=0.0,
                p95_latency_us=0.0,
                p99_latency_us=0.0,
                passed_count=0,
                review_count=0,
                escalated_count=0,
            )

        reports: List[FraudScoringReport] = []
        latencies_us: List[float] = []
        passed = 0
        review = 0
        escalated = 0

        start_time = time.perf_counter()
        for tx in transactions:
            rep = self.scorer.score_transaction(tx)
            latencies_us.append(rep.scoring_latency_us)

            if rep.decision == "PASS":
                passed += 1
            elif rep.decision == "REVIEW":
                review += 1
            else:
                escalated += 1

            if capture_individual_reports:
                reports.append(rep)

        total_duration = time.perf_counter() - start_time
        total_duration_ms = total_duration * 1000.0
        throughput = total / max(1e-8, total_duration)

        lat_arr = np.array(latencies_us)

        return ReplayBatchResult(
            total_transactions=total,
            throughput_tx_per_sec=round(throughput, 1),
            total_duration_ms=round(total_duration_ms, 2),
            mean_latency_us=round(float(np.mean(lat_arr)), 2),
            p50_latency_us=round(float(np.percentile(lat_arr, 50)), 2),
            p95_latency_us=round(float(np.percentile(lat_arr, 95)), 2),
            p99_latency_us=round(float(np.percentile(lat_arr, 99)), 2),
            passed_count=passed,
            review_count=review,
            escalated_count=escalated,
            reports=reports,
        )
