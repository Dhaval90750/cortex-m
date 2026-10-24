"""SQL & Database Optimization Skill for Cortex-M & CortexCode.

Analyzes SQL queries and table schemas for:
1. `SELECT *` table-scan anti-patterns.
2. Non-SARGable WHERE predicates (e.g. function calls on indexed columns).
3. Missing covering indexes on JOIN / WHERE columns.
4. Relational vs LSM storage engine suitability recommendations.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from .base import BaseSkill, SkillResult


class SqlOptimizationSkill(BaseSkill):
    """Analyzes SQL queries and schemas for index optimization and performance."""

    def __init__(self):
        super().__init__(
            name="optimize_sql",
            description="Analyzes SQL queries for missing indexes, non-SARGable clauses, and storage engine suitability.",
            category="database",
        )

    def execute(self, target: str, context: Optional[Dict[str, Any]] = None) -> SkillResult:
        t0 = time.perf_counter()
        target_clean = target.strip()
        target_lower = target_clean.lower()
        findings: List[str] = []
        recommendations: List[str] = []

        # 1. SELECT * detection
        if re.search(r"\bselect\s+\*\s+from\b", target_lower):
            findings.append("[WARNING] 'SELECT *' detected. Fetches unnecessary columns and bypasses covering index scans.")
            recommendations.append("Explicitly specify required column projections (e.g. 'SELECT id, created_at FROM ...').")

        # 2. Leading wildcard LIKE '%pattern'
        if re.search(r"like\s+['\"]%.*?['\"]", target_lower):
            findings.append("[HIGH] Leading wildcard in LIKE expression (e.g. '%abc') forces full sequential table scan.")
            recommendations.append("Use full-text search (GIN index in PostgreSQL) or prefix matching ('abc%').")

        # 3. Non-SARGable function calls in WHERE
        if re.search(r"where\s+[a-z0-9_]+\s*\([a-z0-9_]+\)", target_lower):
            findings.append("[MEDIUM] Function call wrapping column in WHERE clause (non-SARGable predicate).")
            recommendations.append("Rewrite predicate to compare raw column against evaluated expression or create expression index.")

        # 4. Extract WHERE / JOIN columns for index suggestions
        where_match = re.search(r"\bwhere\s+([a-zA-Z0-9_\s=><!]+)", target_lower)
        if where_match:
            cols = re.findall(r"([a-zA-Z0-9_]+)\s*=", where_match.group(1))
            if cols:
                recommendations.append(f"Create compound B-Tree index on filter columns: CREATE INDEX idx_composite ON table_name ({', '.join(cols)});")

        # 5. Storage Engine Suitability
        if "insert" in target_lower or "append" in target_lower:
            recommendations.append("For write-heavy log workloads, use LSM-Tree storage (e.g. RocksDB/Cassandra) instead of B-Tree.")
        elif "select" in target_lower and "join" in target_lower:
            recommendations.append("For complex relational joins, B-Tree indexed Postgres/MySQL remains optimal.")

        dt = (time.perf_counter() - t0) * 1000.0
        success = len(findings) == 0

        report = "SQL Optimization Report:\n"
        if findings:
            report += "Identified Bottlenecks:\n" + "\n".join(f"  * {f}" for f in findings) + "\n\n"
        else:
            report += "Query Structure: Clean. Zero anti-patterns detected.\n\n"

        if recommendations:
            report += "Optimization Guidelines:\n" + "\n".join(f"  - {r}" for r in recommendations)

        return SkillResult(
            skill_name=self.name,
            success=success,
            output=report,
            metadata={"findings_count": len(findings), "has_sargable_issues": bool(findings)},
            execution_time_ms=round(dt, 2),
            resonance_score=0.91,
            findings=findings,
            suggestions=recommendations,
        )
