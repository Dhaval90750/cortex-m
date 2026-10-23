from __future__ import annotations

from collections.abc import Iterable

from .memory import CortexMemory
from .render import render_response
from .rules import RuleFactExtractor
from .store import MemoryStore
from .types import ResponsePlan


class RuleBasedCortex:
    """Runnable no-model memory assistant with a caller-configured lexicon."""

    def __init__(self, keys: Iterable[str], values: Iterable[str],
                 store: MemoryStore | None = None) -> None:
        self.extractor = RuleFactExtractor(keys, values)
        self.store = store or MemoryStore()
        self.memory = CortexMemory(self.store)
        self.turn_index = 0

    def observe(self, user_text: str) -> int:
        """Extract and store only lexicon-matched, explicitly stated facts."""
        self.turn_index += 1
        return self.store.ingest(self.extractor.extract(user_text, self.turn_index))

    def plan(self, query: str) -> ResponsePlan:
        _, _, response_plan = self.memory.prepare(query, goal=query)
        return response_plan

    def respond(self, query: str) -> str:
        return render_response(self.plan(query))

    def close(self) -> None:
        self.store.close()
