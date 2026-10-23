from __future__ import annotations

import uuid
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
