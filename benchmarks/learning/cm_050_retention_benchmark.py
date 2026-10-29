"""Forwarding wrapper for CM-050 Continual Learning Benchmark.

Enables both path invocations:
- benchmarks/run_continual_learning_benchmark.py
- benchmarks/learning/cm_050_retention_benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from cortex_m.learning.continual import CM050RetentionBenchmark

SEEDS = [42, 137, 256, 512, 1024]


def main() -> None:
    results = CM050RetentionBenchmark.run_full_benchmark(seeds=SEEDS)
    gate_passed = results["metadata"]["gate_passed"]
    print(f"\n========================================================")
    print(f" CM-050 Benchmark Complete | Gate Status: {'PASSED' if gate_passed else 'FAILED'}")
    print(f"========================================================\n")
    if not gate_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
