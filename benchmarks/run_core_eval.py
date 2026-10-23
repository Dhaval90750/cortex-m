from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from cortex_m.extractor import AnnotatedFactExtractor
from cortex_m.memory import CortexMemory
from cortex_m.store import MemoryStore

ROOT = Path(__file__).resolve().parents[1]


def evaluate(case: dict) -> dict:
    store = MemoryStore()
    if case["memory_outage"]:
        store.available = False
    else:
        store.ingest(AnnotatedFactExtractor().extract(case["annotations"]))
    start = time.perf_counter()
    retrieval, gist, plan = CortexMemory(store).prepare(case["query"], goal=case["query"])
    elapsed = time.perf_counter() - start
    if case["gold_action"] == "answer":
        values = [item.fact.value for item in plan.evidence]
        success = plan.action == "answer" and case["gold_value"] in values
    elif case["gold_action"] == "clarify":
        values = {item.fact.value for item in retrieval.evidence}
        expected = {item["value"] for item in case["annotations"]}
        success = plan.action == "clarify" and expected <= values
    else:
        success = plan.action == case["gold_action"]
    record = {"id": case["id"], "category": case["category"],
              "expected_action": case["gold_action"], "actual_action": plan.action,
              "expected_value": case["gold_value"],
              "retrieved_values": [e.fact.value for e in retrieval.evidence],
              "retrieval_status": retrieval.status,
              "evidence_count": len(retrieval.evidence),
              "gist_approx_tokens": gist.approximate_tokens,
              "latency_seconds": elapsed, "success": success}
    store.close()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the earlier annotation-fed core; separate from the model-free four-way phrase-rule benchmark.")
    parser.add_argument("--split", choices=("dev", "holdout"), default="dev")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data_path = ROOT / "data" / "generated" / f"{args.split}.jsonl"
    cases = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [evaluate(case) for case in cases]
    run_id = datetime.now(timezone.utc).strftime("core-%Y%m%dT%H%M%S%fZ")
    out = args.output or ROOT / "results" / run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps({"split": args.split, "system": "Cortex-M deterministic core", "language_model": False, "fact_source": "benchmark annotations", "case_count": len(cases)}, indent=2) + "\n", encoding="utf-8")
    (out / "environment.json").write_text(json.dumps({"python": sys.version, "platform": platform.platform(), "implementation": platform.python_implementation()}, indent=2) + "\n", encoding="utf-8")
    (out / "dataset_manifest.json").write_text((ROOT / "data" / "generated" / "manifest.json").read_text(encoding="utf-8"), encoding="utf-8")
    (out / "case_results.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    summary = {"run_id": run_id, "split": args.split, "n": len(rows),
               "successes": sum(row["success"] for row in rows),
               "exact_success_rate": statistics.mean(row["success"] for row in rows),
               "by_category": {category: {"n": sum(row["category"] == category for row in rows),
                                            "success_rate": statistics.mean(row["success"] for row in rows if row["category"] == category)}
                               for category in sorted({row["category"] for row in rows})},
               "p95_core_latency_seconds": sorted(row["latency_seconds"] for row in rows)[max(0, int(0.95 * len(rows)) - 1)],
               "interpretation": "Deterministic component check using gold annotations; not a language-model or baseline comparison."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "stdout.log").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "interpretation.md").write_text(
        "# Deterministic core evaluation\n\n"
        f"Split: `{args.split}`. Cases: {len(rows)}. Successes: {summary['successes']}.\n\n"
        "The benchmark supplies facts through the annotation extractor to isolate storage, retrieval, planning, and outage behavior. It does not measure language-model extraction or generation, and it does not establish an advantage over transcript baselines. Those phases require the gated local-model run.\n",
        encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
