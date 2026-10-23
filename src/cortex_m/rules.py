from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from .types import FactRecord


class RuleFactExtractor:
    """No-model extractor for explicitly stated facts in a configured vocabulary.

    This is a bounded rule system, not general language understanding. Unknown
    keys/values are left unextracted so callers can ask for clarification.
    """

    NEGATION_MARKERS = (
        "never told", "not told", "haven't told", "have not told",
        "never gave", "not given", "not stated", "not supplied",
        "no personal", "didn't say", "did not say",
    )
    NONASSERTIVE_MARKERS = (
        "maybe", "perhaps", "might be", "could be", "i guess", "i think",
        "i am not sure", "i'm not sure", "i don't know", "i do not know",
        "someone else", "my friend said", "they said", "hypothetically",
        "didn't confirm", "did not confirm",
    )
    ASSERTION_MARKERS = (
        "i prefer", "i told you", "you told me", "i previously stated",
        "i stated", "i said", "i explicitly chose", "please remember",
        "my ", "my choice", "my statements", "change my", "correct current",
        "correction:", "set ", "revise ", "replace ", "the old ",
        "i used to say", "update it to", "for ",
    )
    CORRECTION_MARKERS = (
        "correction", "update", "change", "revise", "replace",
        "correct current", "from ",
    )

    def __init__(self, keys: Iterable[str], values: Iterable[str]) -> None:
        self.keys = tuple(sorted(set(keys), key=len, reverse=True))
        self.values = tuple(sorted(set(values), key=len, reverse=True))
        self._value_patterns = {
            value: re.compile(r"(?<!\w)" + re.escape(value) + r"(?!\w)", re.I)
            for value in self.values
        }

    @staticmethod
    def _contains(text: str, phrase: str) -> bool:
        return bool(re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text, re.I))

    def _found_values(self, text: str) -> list[tuple[int, str]]:
        found = []
        for value, pattern in self._value_patterns.items():
            found.extend((match.start(), match.end(), value)
                         for match in pattern.finditer(text))
        found.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        non_overlapping = []
        for candidate in found:
            if any(candidate[0] < item[1] and item[0] < candidate[1]
                   for item in non_overlapping):
                continue
            non_overlapping.append(candidate)
        return [(start, value) for start, _, value in
                sorted(non_overlapping, key=lambda item: item[0])]

    def extract(self, text: str, turn_index: int) -> list[FactRecord]:
        lowered = text.casefold()
        if any(marker in lowered for marker in self.NEGATION_MARKERS):
            return []
        if any(marker in lowered for marker in self.NONASSERTIVE_MARKERS):
            return []
        found_values = self._found_values(text)
        if not found_values:
            return []
        if not any(marker in lowered for marker in self.ASSERTION_MARKERS):
            return []

        matched_keys = [key for key in self.keys if self._contains(text, key)]
        # Prefer the most specific key when its words also occur in a broader key.
        if matched_keys:
            longest = max(map(len, matched_keys))
            matched_keys = [key for key in matched_keys if len(key) == longest]

        records: list[FactRecord] = []
        is_correction = any(marker in lowered for marker in self.CORRECTION_MARKERS)
        for key in matched_keys:
            associated = found_values
            if not associated:
                continue
            selected = associated
            if is_correction:
                # This exact pattern handles "set key to NEW, replacing OLD".
                value_alt = "|".join(re.escape(value) for value in self.values)
                set_pattern = re.compile(
                    r"set\s+" + re.escape(key) + r"\s+to\s+(" + value_alt + r")\b",
                    re.I,
                )
                set_match = set_pattern.search(text)
                if set_match:
                    selected = [(set_match.start(1), set_match.group(1))]
                else:
                    correct_pattern = re.compile(
                        r"correct current value is\s+(" + value_alt + r")\b", re.I
                    )
                    correct_match = correct_pattern.search(text)
                    if correct_match:
                        selected = [(correct_match.start(1), correct_match.group(1))]
                    else:
                        selected = [associated[-1]]

            seen = set()
            for _, value in selected:
                value = next((v for v in self.values if v.casefold() == value.casefold()), value)
                if value.casefold() in seen:
                    continue
                seen.add(value.casefold())
                stable = hashlib.sha256(
                    f"{turn_index}\0{key}\0{value}\0{text}".encode("utf-8")
                ).hexdigest()[:24]
                records.append(FactRecord(
                    fact_id=stable, key=key, value=value,
                    source_turn=turn_index, evidence_text=text,
                    explicitness="explicit", confidence=1.0,
                    correction=is_correction,
                ))
        return records
