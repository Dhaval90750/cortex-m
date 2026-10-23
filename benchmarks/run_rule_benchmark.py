from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from cortex_m.memory import CortexMemory
from cortex_m.render import render_response
from cortex_m.rules import RuleFactExtractor
from cortex_m.store import MemoryStore
from cortex_m.types import FactRecord

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ("current_turn", "full_transcript", "gist_only", "cortex_m")
LEXICON = json.loads((ROOT / "data" / "lexicon.json").read_text(encoding="utf-8"))


def extract_history(case: dict, extractor: RuleFactExtractor) -> list[FactRecord]:
    facts = []
    for turn_no, turn in enumerate(case["turns"], start=1):
        if turn["role"] == "user":
            facts.extend(extractor.extract(turn["text"], turn_no))
    return facts


def validate_case_annotations(case: dict, extractor: RuleFactExtractor) -> bool:
    observed = extract_history(case, extractor)
    got = sorted((f.key, f.value, f.source_turn, f.correction) for f in observed)
    expected = sorted((f["key"], f["value"], int(f["source_turn"]),
                       bool(f.get("correction", False)))
                      for f in case["annotations"])
    return got == expected


def resolve_facts(facts: list[FactRecord], key: str,
                  unavailable: bool = False) -> tuple[str, str, list[FactRecord]]:
    if unavailable:
        return "fallback", "", []
    matching = [fact for fact in facts
                if " ".join(fact.key.casefold().split()) ==
                " ".join(key.casefold().split())]
    if not matching:
        return "abstain", "", []
    corrections = [fact for fact in matching if fact.correction]
    if corrections:
        return "answer", corrections[-1].value, corrections[-1:]
    distinct = list(dict.fromkeys(fact.value for fact in matching))
    if len(distinct) > 1:
        return "clarify", "", matching
    return "answer", distinct[0], matching


def make_text(action: str, key: str, value: str) -> str:
    if action == "fallback":
        return "Memory is unavailable. Please repeat that detail."
    if action == "abstain":
        return "I don't have that detail recorded, so I won't guess."
    if action == "clarify":
        return "I have conflicting details. Which value should I use?"
    return f"Your recorded {key} is {value}."


def evaluate(case: dict, system: str, extractor: RuleFactExtractor) -> dict:
    start = time.perf_counter()
    query = case["query"]
    key = case["query_key"]
    outage = bool(case["memory_outage"])
    current_user_text = next((turn["text"] for turn in reversed(case["turns"])
                              if turn["role"] == "user"), "")
    if system == "current_turn":
        facts = extractor.extract(current_user_text, len(case["turns"]))
        extracted_count = len(facts)
        action, value, evidence = resolve_facts(facts, key, outage)
        context_words = len(current_user_text.split()) + len(query.split())
    else:
        all_facts = extract_history(case, extractor)
        extracted_count = len(all_facts)
    if system == "current_turn":
        output = make_text(action, key, value)
    elif system == "full_transcript":
        action, value, evidence = resolve_facts(all_facts, key, outage)
        context_words = sum(len(turn["text"].split()) for turn in case["turns"]) + len(query.split())
    elif system == "gist_only":
        gist = {}
        for fact in all_facts:
            gist[fact.key] = fact  # keep only the latest value; no conflict history
        facts = list(gist.values())
        action, value, evidence = resolve_facts(facts, key, outage)
        context_words = len(query.split()) + sum(len(f.key.split()) + len(f.value.split()) for f in facts)
    elif system == "cortex_m":
        store = MemoryStore()
        if outage:
            store.available = False
        else:
            store.ingest(all_facts)
        memory = CortexMemory(store)
        retrieval, gist, plan = memory.prepare(query, goal=query)
        action = plan.action
        evidence = [item.fact for item in plan.evidence]
        value = evidence[0].value if action == "answer" and evidence else ""
        context_words = gist.approximate_tokens
        output = render_response(plan)
        store.close()
        elapsed = time.perf_counter() - start
        return score_record(case, system, action, value, evidence, output,
                            context_words, elapsed, extracted_count)
    else:
        raise ValueError(f"unknown system: {system}")
    output = make_text(action, key, value)
    elapsed = time.perf_counter() - start
    return score_record(case, system, action, value, evidence, output,
                        context_words, elapsed, extracted_count)


def score_record(case, system, action, value, evidence, output, context_words,
                 elapsed, extracted_count):
    correct = action == case["gold_action"] and (
        action != "answer" or value == case["gold_value"])
    unsupported = action == "answer" and value != case["gold_value"]
    return {"id": case["id"], "category": case["category"], "system": system,
            "expected_action": case["gold_action"], "actual_action": action,
            "expected_value": case["gold_value"], "actual_value": value,
            "output": output, "evidence_values": [item.value for item in evidence],
            "correct": correct, "unsupported_answer": unsupported,
            "context_words_approx": context_words,
            "extracted_fact_count": extracted_count,
            "latency_seconds": elapsed}


def paired_bootstrap(rows: list[dict], rng_seed=20261023, repetitions=10000) -> dict:
    differences = []
    for case_id in sorted({row["id"] for row in rows}):
        by_system = {row["system"]: row for row in rows if row["id"] == case_id}
        differences.append(int(by_system["cortex_m"]["correct"]) -
                           int(by_system["full_transcript"]["correct"]))
    rng = random.Random(rng_seed)
    n = len(differences)
    samples = sorted(statistics.mean(rng.choices(differences, k=n))
                     for _ in range(repetitions))
    return {"cortex_minus_full_transcript": statistics.mean(differences),
            "ci95_percentile": [samples[int(0.025 * repetitions)],
                                samples[min(repetitions - 1, int(0.975 * repetitions))]],
            "pairs": n, "bootstrap_repetitions": repetitions, "seed": rng_seed}


def summarize(rows: list[dict], split: str) -> dict:
    by_system = {}
    for system in SYSTEMS:
        subset = [row for row in rows if row["system"] == system]
        by_system[system] = {
            "n": len(subset),
            "exact_task_success": statistics.mean(row["correct"] for row in subset),
            "unsupported_answer_rate": statistics.mean(row["unsupported_answer"] for row in subset),
            "mean_context_words_approx": statistics.mean(row["context_words_approx"] for row in subset),
            "p95_latency_seconds": sorted(row["latency_seconds"] for row in subset)[max(0, int(0.95 * len(subset)) - 1)],
            "by_category": {cat: statistics.mean(row["correct"] for row in subset if row["category"] == cat)
                            for cat in sorted({row["category"] for row in subset})},
        }
    full_words = by_system["full_transcript"]["mean_context_words_approx"]
    cortex_words = by_system["cortex_m"]["mean_context_words_approx"]
    return {"split": split, "cases": len({row["id"] for row in rows}),
            "systems": by_system,
            "cortex_context_word_reduction_vs_full": 1.0 - cortex_words / full_words,
            "paired_bootstrap_cortex_vs_full": paired_bootstrap(rows),
            "method_note": "All four systems use explicit vocabulary/phrase rules and fixed response templates. No AI model, API, network, or case annotations are used for extraction, retrieval, or response generation."}


def main() -> None:
    parser = argparse.ArgumentParser(description="Model-free Cortex-M comparison using rules, exact evidence, and fixed replies.")
    parser.add_argument("--split", choices=("dev", "holdout"), default="dev")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data_path = ROOT / "data" / "generated_v2" / f"{args.split}.jsonl"
    cases = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    extractor = RuleFactExtractor(LEXICON["keys"], LEXICON["values"])
    invalid = [case["id"] for case in cases
               if not validate_case_annotations(case, extractor)]
    if invalid:
        raise SystemExit("Rule extractor/corpus annotation mismatch: " + ", ".join(invalid[:12]))
    rows = [evaluate(case, system, extractor) for case in cases for system in SYSTEMS]
    run_id = datetime.now(timezone.utc).strftime("rule-%Y%m%dT%H%M%S%fZ")
    out = args.output or ROOT / "results" / run_id
    out.mkdir(parents=True, exist_ok=False)
    summary = summarize(rows, args.split)
    summary["run_id"] = run_id
    summary["dataset_manifest"] = "data/generated_v2/manifest.json"
    (out / "config.json").write_text(json.dumps({"split": args.split, "systems": SYSTEMS, "model": None, "api": None, "case_count": len(cases)}, indent=2) + "\n", encoding="utf-8")
    (out / "environment.json").write_text(json.dumps({"python": sys.version, "platform": platform.platform(), "network_inference": False}, indent=2) + "\n", encoding="utf-8")
    (out / "dataset_manifest.json").write_text((ROOT / "data" / "generated_v2" / "manifest.json").read_text(encoding="utf-8"), encoding="utf-8")
    (out / "case_results.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "stdout.log").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "interpretation.md").write_text(
        "# Model-free rule benchmark\n\n"
        "All conditions use deterministic phrase rules and fixed response templates. "
        "Facts are extracted from the conversation text; annotations are used only for scoring.\n\n"
        + "```json\n" + json.dumps(summary, indent=2) + "\n```\n",
        encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
