from __future__ import annotations

"""Run independent end-to-end behavior checks and evaluate saved benchmark gates."""

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from cortex_m.assistant import RuleBasedCortex
from cortex_m.store import MemoryStore

ROOT = Path(__file__).resolve().parents[1]
LEXICON = json.loads((ROOT / "data" / "lexicon.json").read_text(encoding="utf-8"))


def run_case(name: str, action: Callable[[], tuple[bool, dict]]) -> dict:
    try:
        passed, details = action()
        return {"name": name, "status": "PASS" if passed else "FAIL", **details}
    except Exception as exc:  # report a failed case without hiding later checks
        return {"name": name, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path,
                        help="Benchmark result directory; defaults to the newest saved rule-* result.")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results" / f"acceptance-{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="cortex-m-acceptance-") as temp:
        temp_path = Path(temp)
        cases = [
            run_case("explicit recall", lambda: recall_case()),
            run_case("correction and superseded history", lambda: correction_case()),
            run_case("conflict triggers clarification", lambda: conflict_case()),
            run_case("uncertain statement is rejected", lambda: abstention_case("Maybe my preferred editor is Vim.")),
            run_case("unlisted value is rejected", lambda: abstention_case("My preferred editor is VS Code.")),
            run_case("SQLite persistence after reopen", lambda: persistence_case(temp_path / "persistent.sqlite")),
            run_case("clear removes stored fact", lambda: clear_case()),
            run_case("open-vocabulary name survives persistence and is recalled",
                     lambda: name_case(temp_path / "name.sqlite")),
            run_case("unavailable storage falls back safely", lambda: outage_case()),
            run_case("corrupt database fails closed", lambda: corrupt_case(temp_path / "corrupt.sqlite")),
        ]

    benchmark_dir = args.benchmark_dir or newest_benchmark_dir()
    benchmark = evaluate_benchmark(benchmark_dir)
    all_cases_pass = all(case["status"] == "PASS" for case in cases)
    all_gates_pass = all(gate["status"] == "PASS" for gate in benchmark["gates"])
    overall = "PASS" if all_cases_pass and all_gates_pass else (
        "PARTIAL" if all_cases_pass else "FAIL")
    report = {
        "report_id": out.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "overall": overall,
        "scope": "model-free deterministic Cortex-M functional and saved-benchmark acceptance",
        "ai_used": False,
        "cases": cases,
        "functional_cases_passed": sum(c["status"] == "PASS" for c in cases),
        "functional_cases_total": len(cases),
        "benchmark": benchmark,
        "reproduce": [
            ".\\scripts\\python.ps1 scripts\\walkthrough.py",
            ".\\scripts\\python.ps1 scripts\\check_success.py",
            ".\\scripts\\python.ps1 -m unittest discover -s tests -v",
        ],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print((out / "report.md").read_text(encoding="utf-8"))
    print(f"Saved: {out}")


def assistant(store: MemoryStore | None = None) -> RuleBasedCortex:
    return RuleBasedCortex(LEXICON["keys"], LEXICON["values"], store)


def recall_case() -> tuple[bool, dict]:
    c = assistant()
    try:
        accepted = c.observe("My preferred editor is Vim.")
        plan, response = c.plan("What is my preferred editor?"), c.respond("What is my preferred editor?")
        ok = accepted == 1 and plan.action == "answer" and "Vim" in response
        return ok, {"accepted_facts": accepted, "action": plan.action, "response": response}
    finally: c.close()


def correction_case() -> tuple[bool, dict]:
    c = assistant()
    try:
        c.observe("My preferred editor is Vim.")
        accepted = c.observe("Correction: set preferred editor to Emacs, replacing Vim.")
        history = c.store.facts(include_superseded=True)
        active = c.store.facts()
        response = c.respond("What is my preferred editor?")
        ok = accepted == 1 and [(f.value, f.status) for f in history] == [("Vim", "superseded"), ("Emacs", "active")] and "Emacs" in response
        return ok, {"history": [{"value": f.value, "status": f.status} for f in history], "response": response}
    finally: c.close()


def conflict_case() -> tuple[bool, dict]:
    c = assistant()
    try:
        first = c.observe("I prefer Vim for preferred editor.")
        second = c.observe("I prefer Emacs for preferred editor.")
        plan = c.plan("What is my preferred editor?")
        values = sorted(e.fact.value for e in plan.evidence)
        response = c.respond("What is my preferred editor?")
        ok = first == second == 1 and plan.action == "clarify" and values == ["Emacs", "Vim"]
        return ok, {"accepted_facts": [first, second], "action": plan.action, "conflict_values": values, "response": response}
    finally: c.close()


def abstention_case(statement: str) -> tuple[bool, dict]:
    c = assistant()
    try:
        accepted = c.observe(statement)
        query = "What is my preferred editor?"
        plan, response = c.plan(query), c.respond(query)
        ok = accepted == 0 and plan.action == "abstain" and "won't guess" in response
        return ok, {"statement": statement, "accepted_facts": accepted, "action": plan.action, "response": response}
    finally: c.close()


def persistence_case(path: Path) -> tuple[bool, dict]:
    first = assistant(MemoryStore(path))
    first.observe("My preferred editor is Vim.")
    first.close()
    second = assistant(MemoryStore(path))
    try:
        plan, response = second.plan("What is my preferred editor?"), second.respond("What is my preferred editor?")
        ok = plan.action == "answer" and "Vim" in response
        return ok, {"action_after_reopen": plan.action, "response_after_reopen": response}
    finally: second.close()


def name_case(path: Path) -> tuple[bool, dict]:
    first = assistant(MemoryStore(path))
    try:
        accepted = first.observe("my name is dhaval")
    finally:
        first.close()
    second = assistant(MemoryStore(path))
    try:
        query = "what's my name?"
        plan, response = second.plan(query), second.respond(query)
        ok = (accepted == 1 and plan.action == "answer" and
              bool(plan.evidence) and plan.evidence[0].fact.value == "dhaval" and
              response == "Your recorded name is dhaval.")
        return ok, {"statement": "my name is dhaval", "accepted_facts": accepted,
                    "action_after_reopen": plan.action,
                    "response_after_reopen": response,
                    "stored_value": plan.evidence[0].fact.value if plan.evidence else None}
    finally: second.close()


def clear_case() -> tuple[bool, dict]:
    c = assistant()
    try:
        c.observe("My preferred editor is Vim.")
        deleted = c.store.clear_key("preferred editor")
        plan = c.plan("What is my preferred editor?")
        ok = deleted == 1 and plan.action == "abstain"
        return ok, {"deleted": deleted, "action_after_clear": plan.action}
    finally: c.close()


def outage_case() -> tuple[bool, dict]:
    c = assistant()
    try:
        c.store.available = False
        plan, response = c.plan("What is my preferred editor?"), c.respond("What is my preferred editor?")
        ok = plan.action == "fallback" and "unavailable" in response.casefold()
        return ok, {"action": plan.action, "response": response}
    finally: c.close()


def corrupt_case(path: Path) -> tuple[bool, dict]:
    path.write_bytes(b"not a sqlite database")
    c = assistant(MemoryStore(path))
    try:
        plan = c.plan("What is my preferred editor?")
        ok = not c.store.available and plan.action == "fallback"
        return ok, {"store_available": c.store.available, "action": plan.action}
    finally: c.close()


def newest_benchmark_dir() -> Path:
    candidates = sorted((ROOT / "results").glob("rule-*/summary.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No saved results/rule-*/summary.json benchmark found")
    return candidates[0].parent


def evaluate_benchmark(folder: Path) -> dict:
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cortex, full = summary["systems"]["cortex_m"], summary["systems"]["full_transcript"]
    reduction = summary["cortex_context_word_reduction_vs_full"]
    latency_ratio = cortex["p95_latency_seconds"] / full["p95_latency_seconds"]
    category = cortex["by_category"]
    gates = [
        gate("exact success at least full-transcript baseline", cortex["exact_task_success"] >= full["exact_task_success"],
             cortex["exact_task_success"], full["exact_task_success"], ">="),
        gate("unsupported answer rate <= 2%", cortex["unsupported_answer_rate"] <= 0.02,
             cortex["unsupported_answer_rate"], 0.02, "<="),
        gate("approximate context-word reduction >= 30%", reduction >= 0.30,
             reduction, 0.30, ">="),
        gate("p95 latency ratio vs full transcript <= 2.0", latency_ratio <= 2.0,
             latency_ratio, 2.0, "<="),
        gate("recall, correction, conflict, unknown each >= 95%", all(category.get(k, 0) >= 0.95 for k in ("recall", "correction", "conflict", "unknown")),
             category, 0.95, "each >="),
    ]
    return {"source": str(summary_path), "run_id": summary.get("run_id"),
            "split": summary["split"], "cases": summary["cases"],
            "cortex_m_exact_success": cortex["exact_task_success"],
            "cortex_m_unsupported_rate": cortex["unsupported_answer_rate"],
            "context_word_reduction_vs_full": reduction,
            "p95_latency_ratio_vs_full": latency_ratio,
            "category_success": category, "gates": gates}


def gate(name: str, passed: bool, observed: object, target: object, operator: str) -> dict:
    return {"name": name, "status": "PASS" if passed else "FAIL",
            "observed": observed, "target": target, "operator": operator}


def render_markdown(report: dict) -> str:
    lines = [f"# Cortex-M acceptance report: {report['overall']}", "",
             f"Created: {report['created_utc']}", "",
             "## Functional end-to-end cases", "",
             "| Case | Result | Details |", "|---|---|---|"]
    for case in report["cases"]:
        details = case.get("response", case.get("error", "See report.json" )).replace("|", "\\|")
        lines.append(f"| {case['name']} | {case['status']} | {details} |")
    b = report["benchmark"]
    lines += ["", "## Saved benchmark gates", "", f"Source: `{b['source']}`", "",
              "| Gate | Result | Observed | Target |", "|---|---|---:|---:|"]
    for item in b["gates"]:
        lines.append(f"| {item['name']} | {item['status']} | `{json.dumps(item['observed'])}` | {item['operator']} `{json.dumps(item['target'])}` |")
    lines += ["", "## Interpretation", ""]
    if report["overall"] == "PARTIAL":
        lines.append("Functional scenarios passed, but at least one predefined research gate did not. The implementation is functionally demonstrated on these cases; comparative acceptance is partial.")
    elif report["overall"] == "PASS":
        lines.append("All functional scenarios and saved benchmark gates passed for this bounded model-free evaluation.")
    else:
        lines.append("At least one functional end-to-end scenario failed. Inspect the JSON details before claiming functional success.")
    lines += ["", "This check uses deterministic rules and fixed response templates; it does not call an AI model, external API, or network. Context-word reduction is a proxy, not measured model-token savings. The benchmark is a controlled synthetic evaluation, not a general language-understanding result.", "",
              "## Reproduce", "", "```powershell", *report["reproduce"], "```", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
