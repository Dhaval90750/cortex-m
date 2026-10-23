from __future__ import annotations

import json
import re
import uuid

from .backend import LlamaCppBackend
from .types import FactRecord


class AnnotatedFactExtractor:
    """Uses benchmark annotations to isolate memory from language extraction."""

    def extract(self, annotated: list[dict]) -> list[FactRecord]:
        return [FactRecord(
            fact_id=item.get("fact_id") or str(uuid.uuid4()), key=item["key"],
            value=item["value"], source_turn=int(item["source_turn"]),
            evidence_text=item["evidence_text"],
            explicitness=item.get("explicitness", "explicit"),
            confidence=float(item.get("confidence", 1.0)),
            correction=bool(item.get("correction", False)),
        ) for item in annotated]


class ModelFactExtractor:
    """Constrained JSON extraction with exact evidence-span validation."""

    def __init__(self, backend: LlamaCppBackend) -> None:
        self.backend = backend

    def extract_turn(self, text: str, turn_index: int) -> list[FactRecord]:
        messages = [
            {"role": "system", "content": (
                "Extract only facts the user explicitly states about themself. "
                "Return JSON array with key,value,evidence_text,correction. "
                "Do not infer. Return [] if none. Output JSON only. /no_think"
            )},
            {"role": "user", "content": text},
        ]
        result = self.backend.generate(messages, max_tokens=160, temperature=0.1)
        raw = result.text.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
        try:
            values = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"extractor did not return JSON: {raw[:200]}") from exc
        if not isinstance(values, list):
            raise ValueError("extractor output must be a JSON array")
        facts = []
        for item in values:
            if not isinstance(item, dict) or not {"key", "value", "evidence_text"} <= item.keys():
                raise ValueError("fact item is missing a required field")
            evidence = str(item["evidence_text"])
            if not evidence or evidence.casefold() not in text.casefold():
                continue  # reject unsupported or paraphrased evidence spans
            value = str(item["value"])
            if not value or value.casefold() not in evidence.casefold():
                continue
            facts.append(FactRecord(
                fact_id=str(uuid.uuid4()), key=str(item["key"]), value=value,
                source_turn=turn_index, evidence_text=evidence,
                explicitness="explicit", confidence=1.0,
                correction=bool(item.get("correction", False)),
            ))
        return facts
