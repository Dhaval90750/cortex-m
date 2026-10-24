from __future__ import annotations

"""Measure SQLite fact-store and retrieval behavior at a repeatable row count."""

import json
import platform
import sqlite3
import sys
import tempfile
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cortex_m.memory import CortexMemory  # noqa: E402
from cortex_m.store import MemoryStore  # noqa: E402
from cortex_m.types import FactRecord  # noqa: E402


def main() -> None:
    row_count = 10_000
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "results" / f"scale-probe-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    tracemalloc.start()
    with tempfile.TemporaryDirectory(prefix="cortex-m-scale-") as temp:
        db = Path(temp) / "scale.sqlite3"
        records = [FactRecord(
            fact_id=f"scale-{i:05d}", key=f"research detail {i:05d}",
            value=f"value-{i:05d}", source_turn=i + 1,
            evidence_text=(f"Research record {i:05d}: " +
                           "measured detail, source preserved; " * 5),
        ) for i in range(row_count)]
        started = time.perf_counter()
        store = MemoryStore(db)
        accepted = store.ingest(records)
        write_seconds = time.perf_counter() - started
        store.close()
        stored_bytes = db.stat().st_size

        started = time.perf_counter()
        reopened = MemoryStore(db)
        loaded_rows = len(reopened.facts())
        loaded_seconds = time.perf_counter() - started
        started = time.perf_counter()
        result, gist, plan = CortexMemory(reopened).prepare(
            "research detail 00000", goal="retrieve exact research detail"
        )
        retrieval_seconds = time.perf_counter() - started
        expected = "value-00000"
        actual = result.evidence[0].fact.value if result.evidence else None
        passed = (accepted == row_count and loaded_rows == row_count and
                  plan.action == "answer" and actual == expected and
                  gist.approximate_tokens <= 512)
        reopened.close()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    report = {
        "probe": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "rows_requested": row_count,
        "rows_accepted": accepted,
        "rows_after_database_reopen": loaded_rows,
        "known_key_recalled_after_reopen": actual == expected,
        "action": plan.action,
        "gist_token_budget_respected": gist.approximate_tokens <= 512,
        "write_seconds": write_seconds,
        "reopen_load_seconds": loaded_seconds,
        "retrieval_seconds": retrieval_seconds,
        "sqlite_file_bytes": stored_bytes,
        "python_traced_peak_bytes": peak_bytes,
        "status": "PASS" if passed else "FAIL",
        "limits": [
            "Synthetic fact rows inserted directly into SQLite; this does not test natural-language extraction of arbitrary long documents.",
            "One local run is a scale probe, not a production capacity or latency guarantee.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [f"# Cortex-M scale probe: {report['status']}", "",
             f"Created: {report['created_utc']}", "",
             f"- Requested/stored/reopened facts: {row_count}/{accepted}/{loaded_rows}",
             f"- Exact-key recall after reopen: {actual == expected} ({actual})",
             f"- Write: {write_seconds:.3f} s; reopen and load: {loaded_seconds:.3f} s; retrieval: {retrieval_seconds:.3f} s",
             f"- SQLite file: {stored_bytes:,} bytes; traced Python peak: {peak_bytes:,} bytes",
             "", "## Scope limits", "",
             *[f"- {item}" for item in report["limits"]], ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print((output / "report.md").read_text(encoding="utf-8"))
    print(f"Saved: {output}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
