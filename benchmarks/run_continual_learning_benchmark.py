"""CM-050: CORTEX-M Continual Learning and Retention Benchmark Runner.

Governing Standard: CORTEX-M Master Execution Guide (§28, Phase 8, Table 23/24)
Executes multi-seed evaluation across 5 sequential learning conditions:
  1. Naive Fine-Tuning (catastrophic forgetting baseline)
  2. Replay Buffer (M=5 exemplars per task)
  3. Elastic Weight Consolidation (EWC)
  4. Orthogonal Synaptic Grafting (P_perp null-space filter)
  5. EWC + Orthogonal Grafting (Champion)

Generates:
  - results/CM-050_continual_learning.json
  - reports/CM-050_continual_learning.md
  - reports/CM-050_retention_benchmark.md
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
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
