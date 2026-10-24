from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict

from .memory import CortexMemory
from .planner import make_gist, plan_response
from .render import render_response
from .rules import RuleFactExtractor
from .store import MemoryStore, MemoryUnavailableError
from .types import ResponsePlan


class RuleBasedCortex:
    """Runnable no-model memory assistant with a caller-configured lexicon."""

    def __init__(self, keys: Iterable[str], values: Iterable[str],
                 store: MemoryStore | None = None) -> None:
        self.extractor = RuleFactExtractor(keys, values)
        self.store = store or MemoryStore()
        self.memory = CortexMemory(self.store)
        try:
            existing = self.store.facts(include_superseded=True)
            self.turn_index = max((fact.source_turn for fact in existing), default=0)
        except MemoryUnavailableError:
            self.turn_index = 0

    def observe(self, user_text: str) -> int:
        """Extract and store only lexicon-matched, explicitly stated facts."""
        return self.trace_observe(user_text)["accepted_count"]

    def trace_observe(self, user_text: str) -> dict:
        """Run the real observe path and return inspectable execution stages."""
        self.turn_index += 1
        lowered = user_text.casefold()
        matched_keys = [key for key in self.extractor.keys
                        if self.extractor._contains(user_text, key)]
        matched_values = [value for _, value in self.extractor._found_values(user_text)]
        asserted = any(marker in lowered for marker in self.extractor.ASSERTION_MARKERS)
        negated = any(marker in lowered for marker in self.extractor.NEGATION_MARKERS)
        nonassertive = any(marker in lowered for marker in self.extractor.NONASSERTIVE_MARKERS)
        extracted = self.extractor.extract(user_text, self.turn_index)
        matched_values.extend(fact.value for fact in extracted
                              if fact.key == "name" and fact.value not in matched_values)
        stages = [
            {"name": "Input", "status": "done", "detail": "User statement received.",
             "data": {"text": user_text, "turn": self.turn_index}},
            {"name": "Rule scan", "status": "done", "detail": "Configured phrase and value rules inspected the statement.",
             "data": {"matched_keys": matched_keys, "matched_values": matched_values,
                      "assertion_marker_found": asserted, "negation_marker_found": negated,
                      "uncertainty_marker_found": nonassertive}},
            {"name": "Fact extraction", "status": "done" if extracted else "skipped",
             "detail": ("Explicit key/value fact(s) matched." if extracted else
                        "No supported explicit fact was produced; no guess is stored."),
             "data": [asdict(fact) for fact in extracted]},
        ]
        if not extracted:
            if negated:
                reason = "Negation / not-provided rule blocked extraction."
            elif nonassertive:
                reason = "Uncertainty / non-assertive rule blocked extraction."
            elif not matched_values and matched_keys:
                reason = "Key matched, but the value is outside the configured lexicon."
            elif not matched_keys:
                reason = "No configured fact key matched."
            elif not asserted:
                reason = "No explicit assertion marker matched."
            else:
                reason = "The supported phrase pattern did not yield a fact."
            stages.append({"name": "Safety decision", "status": "done",
                           "detail": reason, "data": {"stored": False}})
            return {"kind": "statement", "input": user_text, "stages": stages,
                    "accepted_count": 0, "response": None,
                    "facts": [asdict(f) for f in self.store.facts(include_superseded=True)]}

        before_ids = {fact.fact_id for fact in self.store.facts(include_superseded=True)}
        accepted = self.store.ingest(extracted)
        all_facts = self.store.facts(include_superseded=True)
        changed = [fact for fact in all_facts if fact.fact_id not in before_ids]
        stages += [
            {"name": "SQLite memory", "status": "done",
             "detail": f"{accepted} fact(s) submitted to the persistent memory store.",
             "data": [asdict(fact) for fact in changed]},
            {"name": "Memory state", "status": "done",
             "detail": "Current and historical values are shown with their stored status.",
             "data": [asdict(fact) for fact in all_facts]},
        ]
        return {"kind": "statement", "input": user_text, "stages": stages,
                "accepted_count": accepted, "response": None,
                "facts": [asdict(fact) for fact in all_facts]}

    def plan(self, query: str) -> ResponsePlan:
        _, _, response_plan = self.memory.prepare(query, goal=query)
        return response_plan

    def trace_query(self, query: str) -> dict:
        """Run the real query pipeline and expose each intermediate result."""
        retrieval, gist, response_plan = self.memory.prepare(query, goal=query)
        answer = render_response(response_plan)
        stages = [
            {"name": "Question", "status": "done", "detail": "Question received as the current retrieval goal.",
             "data": {"text": query}},
            {"name": "Memory retrieval", "status": "done" if retrieval.status != "unavailable" else "warning",
             "detail": f"Memory read status: {retrieval.status}.",
             "data": {"status": retrieval.status, "query": retrieval.query}},
            {"name": "Evidence ranking", "status": "done" if retrieval.evidence else "empty",
             "detail": f"{len(retrieval.evidence)} evidence item(s) retrieved.",
             "data": [{"fact": asdict(item.fact), "score": item.score}
                      for item in retrieval.evidence]},
            {"name": "Working gist", "status": "done",
             "detail": "Compact task context assembled from the goal and retrieved evidence.",
             "data": {"goal": gist.current_goal, "topics": gist.active_topics,
                      "evidence_count": len(gist.evidence),
                      "approximate_tokens": gist.approximate_tokens}},
            {"name": "Response plan", "status": "done",
             "detail": response_plan.outline,
             "data": {"action": response_plan.action,
                      "uncertainty": response_plan.uncertainty,
                      "evidence_count": len(response_plan.evidence)}},
            {"name": "Rendered response", "status": "done",
             "detail": "Fixed response template rendered from the response plan.",
             "data": {"text": answer}},
        ]
        try:
            facts = [asdict(fact) for fact in self.store.facts(include_superseded=True)]
        except MemoryUnavailableError:
            facts = []
        return {"kind": "question", "input": query, "stages": stages,
                "action": response_plan.action, "response": answer, "facts": facts}

    def respond(self, query: str) -> str:
        return render_response(self.plan(query))

    def close(self) -> None:
        self.store.close()
