from __future__ import annotations

import re

from .types import ResponsePlan


class EvidenceVerifier:
    """Conservative benchmark verifier; never consults the case's gold label."""

    def verify(self, response: str, plan: ResponsePlan) -> tuple[bool, str]:
        lowered = response.casefold()
        if plan.action == "fallback":
            if any(word in lowered for word in ("unavailable", "repeat", "don't have")):
                return True, response
            return False, "I cannot access prior memory right now; please repeat that detail."
        if plan.action == "abstain":
            if any(word in lowered for word in ("don't know", "not recorded", "no record", "not have")):
                return True, response
            return False, "I don't have that detail recorded, so I won't guess."
        if plan.action == "clarify":
            asks = "?" in response or any(w in lowered for w in ("which", "confirm", "conflict"))
            if asks:
                return True, response
            return False, "I have conflicting details. Which value should I use?"
        allowed = [e.fact.value.casefold() for e in plan.evidence]
        if any(value and value in lowered for value in allowed):
            return True, response
        # If generation omits or contradicts evidence, make a transparent
        # evidence-grounded answer rather than pass through an unsupported one.
        if plan.evidence:
            fact = plan.evidence[0].fact
            return False, f"Your recorded {fact.key.replace('_', ' ')} is {fact.value}."
        return False, "I don't have enough evidence to answer that."

    @staticmethod
    def unsupported_values(response: str, allowed_values: set[str],
                           known_values: set[str]) -> list[str]:
        text = response.casefold()
        return [value for value in known_values - allowed_values
                if re.search(r"(?<!\w)" + re.escape(value.casefold()) + r"(?!\w)", text)]
