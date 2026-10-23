from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 271828
CATEGORIES = ("recall", "correction", "conflict", "unknown")
DEV_TEMPLATES = {
    "recall": ("Earlier you told me your {key} is {value}.", "Please remember: {key} equals {value}.", "My {key} preference: {value}."),
    "correction": ("I used to say {key} was {old}, but update it to {value}.", "Correction: set {key} to {value}, replacing {old}.", "Change my {key} from {old} to {value}."),
    "conflict": ("I prefer {old} for {key}.", "Actually, I prefer {value} for {key}.", "I told you my {key} was {old}, then I said it was {value}."),
    "unknown": ("We discussed {key} generally, but I never gave my own value.", "I have not told you my {key}.", "No personal {key} value was stated."),
}
TEST_TEMPLATES = {
    "recall": ("Recall my {key}: I explicitly chose {value}.", "My recorded {key} should be {value}.", "For {key}, remember {value} as my choice."),
    "correction": ("Please revise {key} from {old} to {value}.", "The old {key} was {old}; the correct current value is {value}.", "Replace my previous {key} ({old}) with {value}."),
    "conflict": ("I previously stated {key}={old}.", "In another message I stated {key}={value}.", "My statements about {key} disagree: {old} and {value}."),
    "unknown": ("Can you infer my personal {key} from our chat?", "What is my {key}? I have not stated it.", "Tell me my {key}, although no personal value was supplied."),
}
QUERIES = {
    "recall": "What is my {key}?",
    "correction": "What is my current {key}?",
    "conflict": "What is my {key}?",
    "unknown": "What is my personal {key}?",
}
LEXICON = json.loads((ROOT / "data" / "lexicon.json").read_text(encoding="utf-8"))
KEYS = tuple(LEXICON["keys"])
VALUES = tuple(LEXICON["values"][:len(KEYS)])
OLD_VALUES = tuple(LEXICON["values"][len(KEYS):])
DEV_HISTORY = (
    ("I organized the draft outline.", "The draft is ready for another review."),
    ("I checked the calendar this morning.", "The schedule is saved for later."),
    ("I sorted several project notes.", "Those notes are ready to revisit."),
    ("I reviewed the open task list.", "The task list remains available."),
)
HOLDOUT_HISTORY = (
    ("I updated my reading notes.", "The reading notes are filed."),
    ("I checked the project timeline.", "The timeline is available for review."),
    ("I sorted the reference documents.", "The documents are ready to use."),
    ("I reviewed the current work queue.", "The queue is saved for next time."),
)


def build_case(split: str, category: str, index: int, rng: random.Random) -> dict:
    # Split-specific paraphrases and distinct fact values prevent template leakage.
    pool = DEV_TEMPLATES if split == "dev" else TEST_TEMPLATES
    slot = index % len(KEYS)
    key = KEYS[slot]
    offset = 0 if split == "dev" else 1
    value = VALUES[(slot + offset) % len(VALUES)]
    old = OLD_VALUES[(slot + offset) % len(OLD_VALUES)]
    turns: list[dict] = []
    annotations: list[dict] = []
    if category == "unknown":
        turns.append({"role": "user", "text": rng.choice(pool[category]).format(key=key)})
        action, gold = ("fallback", "") if index % 5 == 0 else ("abstain", "")
        if action == "fallback":
            turns[-1]["memory_unavailable"] = True
    elif category == "conflict":
        for n, fact_value in enumerate((old, value), start=1):
            text = pool[category][n - 1].format(key=key, old=old, value=value)
            turns.append({"role": "user", "text": text})
            annotations.append({"fact_id": f"{split}-{category}-{index}-{n}", "key": key, "value": fact_value, "source_turn": n, "evidence_text": text, "explicitness": "explicit"})
        action, gold = "clarify", ""
    elif category == "correction":
        text = pool[category][index % 3].format(key=key, old=old, value=value)
        turns.append({"role": "user", "text": text})
        annotations.append({"fact_id": f"{split}-{category}-{index}-1", "key": key, "value": value, "source_turn": 1, "evidence_text": text, "explicitness": "explicit", "correction": True})
        action, gold = "answer", value
    else:
        text = pool[category][index % 3].format(key=key, value=value)
        turns.append({"role": "user", "text": text})
        annotations.append({"fact_id": f"{split}-{category}-{index}-1", "key": key, "value": value, "source_turn": 1, "evidence_text": text, "explicitness": "explicit"})
        action, gold = "answer", value
    fillers = DEV_HISTORY if split == "dev" else HOLDOUT_HISTORY
    # Add harmless conversation turns after the facts. This creates a real
    # retention challenge for current-turn-only systems without model calls.
    history = [{"role": role, "text": text}
               for pair in fillers for role, text in zip(("user", "assistant"), pair)]
    turns.extend(history)
    query = QUERIES[category].format(key=key)
    return {"id": f"{split}-{category}-{index:03d}", "split": split, "category": category,
            "turns": turns, "annotations": annotations, "query": query,
            "query_key": key, "gold_action": action, "gold_value": gold,
            "forbidden_values": [v for v in (value, old) if v != gold],
            "memory_outage": action == "fallback"}


def main() -> None:
    rng = random.Random(SEED)
    cases = []
    for split, count in (("dev", 20), ("holdout", 60)):
        for category in CATEGORIES:
            for i in range(count):
                cases.append(build_case(split, category, i, rng))
    out = ROOT / "data" / "generated_v2"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": SEED, "generator": "data/generate.py", "version": 2,
                "template_pools_disjoint": True,
                "history_turns": 8,
                "conflict_templates_are_explicit_self_statements": True,
                "lexicon_sha256": hashlib.sha256((ROOT / "data" / "lexicon.json").read_bytes()).hexdigest(),
                "counts": {s: {c: count for c in CATEGORIES} for s, count in (("dev", 20), ("holdout", 60))}}
    for split in ("dev", "holdout"):
        rows = [c for c in cases if c["split"] == split]
        path = out / f"{split}.jsonl"
        payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
        path.write_text(payload, encoding="utf-8")
        manifest[f"{split}_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
