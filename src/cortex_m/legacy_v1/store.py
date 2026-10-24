from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Iterable

from .types import FactRecord


class MemoryUnavailableError(RuntimeError):
    pass


class MemoryStore:
    """SQLite event-backed user facts with correction and conflict semantics."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self.available = True
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        try:
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS facts (
                  fact_id TEXT PRIMARY KEY, key_norm TEXT NOT NULL, key TEXT NOT NULL,
                  value TEXT NOT NULL, source_turn INTEGER NOT NULL,
                  evidence_text TEXT NOT NULL, explicitness TEXT NOT NULL,
                  confidence REAL NOT NULL, correction INTEGER NOT NULL,
                  status TEXT NOT NULL, conflict_group TEXT, created_at TEXT
                );
                CREATE INDEX IF NOT EXISTS facts_key_status ON facts(key_norm,status);
                CREATE TABLE IF NOT EXISTS audit_events (
                  event_id TEXT PRIMARY KEY, action TEXT NOT NULL, key_norm TEXT,
                  fact_id TEXT, details TEXT NOT NULL DEFAULT '{}'
                );
                """
            )
            self.connection.commit()
        except sqlite3.DatabaseError:
            # Fail closed. Keep a harmless in-memory connection so close() is
            # safe; retrieval maps unavailable storage to a transparent fallback.
            self.connection.close()
            self.connection = sqlite3.connect(":memory:")
            self.connection.row_factory = sqlite3.Row
            self.available = False

    @staticmethod
    def normalize_key(key: str) -> str:
        return " ".join(key.casefold().replace("_", " ").split())

    def _require_available(self) -> None:
        if not self.available:
            raise MemoryUnavailableError("memory store is unavailable")

    def add_audit(self, action: str, key: str | None = None,
                  fact_id: str | None = None, *, _commit: bool = True,
                  **details: object) -> None:
        self._require_available()
        self.connection.execute(
            "INSERT INTO audit_events VALUES(?,?,?,?,?)",
            (str(uuid.uuid4()), action,
             self.normalize_key(key) if key else None,
             fact_id, json.dumps(details, ensure_ascii=False, sort_keys=True)),
        )
        if _commit:
            self.connection.commit()

    def ingest(self, facts: Iterable[FactRecord]) -> int:
        self._require_available()
        accepted = 0
        with self.connection:
            for fact in facts:
                if fact.explicitness != "explicit":
                    self.add_audit("ignore_inference", fact.key,
                                   fact.fact_id, _commit=False,
                                   source_turn=fact.source_turn)
                    continue
                key_norm = self.normalize_key(fact.key)
                current = self.connection.execute(
                    "SELECT fact_id,value,status FROM facts WHERE key_norm=? "
                    "AND status IN ('active','conflicted')",
                    (key_norm,),
                ).fetchall()
                if fact.correction:
                    self.connection.execute(
                        "UPDATE facts SET status='superseded' WHERE key_norm=? "
                        "AND status IN ('active','conflicted')", (key_norm,),
                    )
                    status, conflict_group = "active", None
                else:
                    distinct_values = {row["value"] for row in current}
                    if distinct_values and fact.value not in distinct_values:
                        conflict_group = str(uuid.uuid4())
                        self.connection.execute(
                            "UPDATE facts SET status='conflicted', conflict_group=? "
                            "WHERE key_norm=? AND status IN ('active','conflicted')",
                            (conflict_group, key_norm),
                        )
                        status = "conflicted"
                    elif any(row["status"] == "conflicted" for row in current):
                        status = "conflicted"
                        conflict_group = next(
                            (row["conflict_group"] for row in self.connection.execute(
                                "SELECT conflict_group FROM facts WHERE key_norm=? "
                                "AND status='conflicted'", (key_norm,)
                            ).fetchall()), None
                        )
                    else:
                        status, conflict_group = "active", None
                self.connection.execute(
                    "INSERT OR IGNORE INTO facts VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (fact.fact_id, key_norm, fact.key, fact.value,
                     fact.source_turn, fact.evidence_text, fact.explicitness,
                     fact.confidence, int(fact.correction), status,
                     conflict_group, fact.created_at),
                )
                self.add_audit("ingest", fact.key, fact.fact_id, _commit=False,
                               status=status, source_turn=fact.source_turn)
                accepted += 1
        return accepted

    def facts(self, include_superseded: bool = False) -> list[FactRecord]:
        self._require_available()
        if include_superseded:
            rows = self.connection.execute(
                "SELECT * FROM facts ORDER BY source_turn,fact_id"
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT * FROM facts WHERE status IN ('active','conflicted') "
                "ORDER BY source_turn,fact_id"
            ).fetchall()
        return [FactRecord(
            fact_id=r["fact_id"], key=r["key"], value=r["value"],
            source_turn=r["source_turn"], evidence_text=r["evidence_text"],
            explicitness=r["explicitness"], confidence=r["confidence"],
            correction=bool(r["correction"]), status=r["status"],
            conflict_group=r["conflict_group"], created_at=r["created_at"],
        ) for r in rows]

    def clear_key(self, key: str) -> int:
        self._require_available()
        key_norm = self.normalize_key(key)
        rows = self.connection.execute(
            "SELECT fact_id FROM facts WHERE key_norm=?", (key_norm,)
        ).fetchall()
        with self.connection:
            self.connection.execute("DELETE FROM facts WHERE key_norm=?", (key_norm,))
            self.add_audit("clear_key", key, _commit=False, count=len(rows))
        return len(rows)

    def clear_all(self) -> int:
        self._require_available()
        count = self.connection.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        with self.connection:
            self.connection.execute("DELETE FROM facts")
            self.add_audit("clear_all", _commit=False, count=count)
        return int(count)

    def close(self) -> None:
        self.connection.close()
