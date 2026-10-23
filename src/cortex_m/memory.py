from __future__ import annotations

from .planner import make_gist, plan_response
from .retrieval import BM25Retriever
from .store import MemoryStore
from .types import FactRecord, RetrievalResult, ResponsePlan, WorkingGist


class CortexMemory:
    def __init__(self, store: MemoryStore | None = None,
                 retriever: BM25Retriever | None = None,
                 token_budget: int = 512) -> None:
        self.store = store or MemoryStore()
        self.retriever = retriever or BM25Retriever()
        self.token_budget = token_budget

    def observe(self, facts: list[FactRecord]) -> int:
        return self.store.ingest(facts)

    def prepare(self, query: str, goal: str | None = None,
                topics: tuple[str, ...] = ()) -> tuple[RetrievalResult, WorkingGist, ResponsePlan]:
        result = self.retriever.retrieve(self.store, query)
        gist = make_gist(goal or query, topics, result,
                         token_budget=self.token_budget)
        plan = plan_response(goal or query, result)
        return result, gist, plan
