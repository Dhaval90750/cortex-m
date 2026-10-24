from __future__ import annotations

"""Show Cortex-M's decisions and the evidence behind each one."""

import argparse
import json
import tempfile
from pathlib import Path

from cortex_m.assistant import RuleBasedCortex
from cortex_m.store import MemoryStore

ROOT = Path(__file__).resolve().parents[1]
LEXICON = json.loads((ROOT / "data" / "lexicon.json").read_text(encoding="utf-8"))


def show_step(title: str, cortex: RuleBasedCortex, statement: str, query: str) -> None:
    accepted = cortex.observe(statement)
    facts = cortex.store.facts(include_superseded=True)
    plan = cortex.plan(query)
    print(f"\n{'=' * 76}\n{title}\n{'=' * 76}")
    print(f"INPUT: {statement}")
    print(f"WRITE: {accepted} fact(s) accepted")
    print("STORE: " + json.dumps([
        {"key": f.key, "value": f.value, "status": f.status,
         "source_turn": f.source_turn, "evidence": f.evidence_text}
        for f in facts
    ], ensure_ascii=False, indent=2))
    print(f"QUERY: {query}")
    print(f"PLAN: action={plan.action}; reason={plan.uncertainty or 'evidence found'}")
    print("EVIDENCE: " + json.dumps([
        {"key": e.fact.key, "value": e.fact.value, "status": e.fact.status,
         "source_turn": e.fact.source_turn, "score": round(e.score, 4)}
        for e in plan.evidence
    ], ensure_ascii=False))
    print("RESPONSE: " + cortex.respond(query))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-db", type=Path, help="Use this SQLite file instead of a temporary demonstration database.")
    args = parser.parse_args()
    if args.keep_db:
        args.keep_db.parent.mkdir(parents=True, exist_ok=True)
        cortex = RuleBasedCortex(LEXICON["keys"], LEXICON["values"], MemoryStore(args.keep_db))
        try:
            _run(cortex)
        finally:
            cortex.close()
    else:
        with tempfile.TemporaryDirectory(prefix="cortex-m-walkthrough-") as temp:
            cortex = RuleBasedCortex(LEXICON["keys"], LEXICON["values"],
                                     MemoryStore(Path(temp) / "walkthrough.sqlite"))
            try:
                _run(cortex)
            finally:
                cortex.close()


def _run(cortex: RuleBasedCortex) -> None:
    print("Cortex-M walkthrough | deterministic rules only | no AI/API/network")
    print("Each section shows the source text, stored state, retrieved evidence, plan, and rendered answer.")
    show_step("1. Explicit fact and provenance", cortex,
              "My preferred editor is Vim.", "What is my preferred editor?")
    show_step("2. Explicit correction", cortex,
              "Correction: set preferred editor to Emacs, replacing Vim.",
              "What is my preferred editor?")
    show_step("3. Unresolved disagreement", cortex,
              "I prefer Vim for preferred editor.", "What is my preferred editor?")
    show_step("4. Uncertain statement is not promoted to fact", cortex,
              "Maybe my usual drink is green tea.", "What is my usual drink?")
    show_step("5. Value outside the configured vocabulary", cortex,
              "My usual drink is oat milk.", "What is my usual drink?")
    print("\n6. Storage outage behavior")
    cortex.store.available = False
    plan = cortex.plan("What is my preferred editor?")
    print(f"PLAN: action={plan.action}; reason={plan.uncertainty}")
    print("RESPONSE: " + cortex.respond("What is my preferred editor?"))


if __name__ == "__main__":
    main()
