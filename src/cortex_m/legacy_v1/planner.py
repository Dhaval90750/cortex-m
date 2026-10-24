from __future__ import annotations

from .types import ResponsePlan, RetrievalResult, WorkingGist


def plan_response(goal: str, retrieval: RetrievalResult) -> ResponsePlan:
    if retrieval.status == "unavailable":
        return ResponsePlan("fallback", goal, uncertainty="memory unavailable",
                            outline="Say prior memory is unavailable; ask the user to repeat the detail.")
    if not retrieval.evidence:
        return ResponsePlan("abstain", goal, uncertainty="no supporting memory",
                            outline="Say the detail is not recorded; do not guess.")
    conflicted = [e for e in retrieval.evidence if e.fact.status == "conflicted"]
    if conflicted:
        return ResponsePlan("clarify", goal, retrieval.evidence,
                            uncertainty="conflicting explicit statements",
                            outline="State that the notes conflict and ask which value is current.")
    return ResponsePlan("answer", goal, retrieval.evidence,
                        outline="Answer briefly using only the retrieved explicit evidence.")


def make_gist(goal: str, topics: tuple[str, ...], retrieval: RetrievalResult,
              unresolved: tuple[str, ...] = (), token_budget: int = 512) -> WorkingGist:
    budget = max(1, token_budget)
    goal_words = goal.split()[:budget]
    goal = " ".join(goal_words)
    remaining = budget - len(goal_words)
    kept_topics = []
    for topic in topics:
        words = topic.split()
        if not words or len(words) > remaining:
            continue
        kept_topics.append(topic)
        remaining -= len(words)
    kept_unresolved = []
    for question in unresolved:
        words = question.split()
        if not words or len(words) > remaining:
            continue
        kept_unresolved.append(question)
        remaining -= len(words)
    evidence = list(retrieval.evidence)
    selected = []
    used = max(1, len(goal_words)) + sum(len(t.split()) for t in kept_topics)
    used += sum(len(q.split()) for q in kept_unresolved)
    for item in evidence:
        cost = max(1, len((item.fact.key + " " + item.fact.value + " " +
                           item.fact.evidence_text).split()))
        if used + cost <= token_budget:
            selected.append(item)
            used += cost
    return WorkingGist(goal, tuple(kept_topics), tuple(selected),
                       tuple(kept_unresolved), used)
