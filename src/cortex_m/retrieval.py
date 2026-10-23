from __future__ import annotations

import math
import re
from collections import Counter

from .store import MemoryStore, MemoryUnavailableError
from .types import EvidenceRef, RetrievalResult

TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold().replace("_", " "))


class BM25Retriever:
    def __init__(self, k1: float = 1.2, b: float = 0.75,
                 min_score: float = 0.05) -> None:
        self.k1, self.b, self.min_score = k1, b, min_score

    def retrieve(self, store: MemoryStore, query: str,
                 limit: int = 8) -> RetrievalResult:
        try:
            facts = store.facts()
        except MemoryUnavailableError:
            return RetrievalResult((), "unavailable", query)
        if not facts:
            return RetrievalResult((), "empty", query)
        q = set(tokens(query))
        if not q:
            return RetrievalResult((), "empty", query)
        normalized_query = " ".join(tokens(query))
        docs = [tokens(f"{f.key} {f.value} {f.evidence_text}") for f in facts]
        df = Counter(term for doc in docs for term in set(doc))
        avg_len = sum(map(len, docs)) / max(len(docs), 1)
        ranked: list[EvidenceRef] = []
        for fact, doc in zip(facts, docs):
            counts = Counter(doc)
            score = 0.0
            for term in q:
                tf = counts[term]
                if not tf:
                    continue
                idf = math.log(1 + (len(docs) - df[term] + 0.5) /
                               (df[term] + 0.5))
                denom = tf + self.k1 * (1 - self.b + self.b *
                                         len(doc) / max(avg_len, 1.0))
                score += idf * tf * (self.k1 + 1) / denom
            normalized_key = self.normalize_key(fact.key)
            if normalized_key and normalized_key in normalized_query:
                score += 1_000_000.0
            if score >= self.min_score:
                ranked.append(EvidenceRef(fact, score))
        ranked.sort(key=lambda item: (-item.score, item.fact.source_turn))
        if not ranked:
            return RetrievalResult((), "empty", query)
        # If the user names a stored key directly, retrieve only that key.
        # This prevents generic words like "my" or hostile evidence text from
        # pulling an unrelated conflict into the response plan.
        exact_keys = [key for key in {self.normalize_key(f.key) for f in facts}
                      if key and key in normalized_query]
        if exact_keys:
            longest = max(map(len, exact_keys))
            exact_keys = [key for key in exact_keys if len(key) == longest]
            ranked = [item for item in ranked
                      if self.normalize_key(item.fact.key) in exact_keys]
        # If a retrieved key is conflicted, include every side of that conflict.
        selected = ranked[:limit]
        conflict_keys = {e.fact.key for e in selected
                         if e.fact.status == "conflicted"}
        for candidate in ranked[limit:]:
            if candidate.fact.key in conflict_keys:
                selected.append(candidate)
        return RetrievalResult(tuple(selected), "ok", query)

    @staticmethod
    def normalize_key(value: str) -> str:
        return " ".join(tokens(value))
