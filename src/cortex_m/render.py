from __future__ import annotations

from .types import ResponsePlan


def render_response(plan: ResponsePlan) -> str:
    """Render a fixed, evidence-constrained reply with no generative model."""
    if plan.action == "fallback":
        return "Memory is unavailable. Please repeat that detail."
    if plan.action == "abstain":
        return "I don't have that detail recorded, so I won't guess."
    if plan.action == "clarify":
        return "I have conflicting details. Which value should I use?"
    fact = plan.evidence[0].fact
    return f"Your recorded {fact.key} is {fact.value}."
