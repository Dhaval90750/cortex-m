"""Skill Registry & Dynamic Dispatch Engine for Cortex-M & CortexCode.

Manages, registers, discovers, and executes modular cognitive & software engineering skills.
"""

from __future__ import annotations

import difflib
from typing import Any, Dict, List, Optional

from .api_scaffolder import ApiScaffoldingSkill
from .base import BaseSkill, SkillResult
from .cognitive_skills import (
    AlignerSkill,
    AskingSkill,
    GoalSkill,
    GrillingSkill,
    PlanningSkill,
    SchedulingSkill,
    SuggestingSkill,
)
from .complexity_analyzer import ComplexityAnalysisSkill
from .doc_generator import DocumentationSkill
from .refactoring import RefactoringSkill
from .security_audit import SecurityAuditSkill
from .sql_optimizer import SqlOptimizationSkill
from .synaptic_skill import SynapticMemorySkill
from .test_generator import TestGenerationSkill


class SkillRegistry:
    """Central registry and associative dispatcher for CORTEX-M skills."""

    def __init__(self, cortex: Optional[Any] = None, completer: Optional[Any] = None):
        self._skills: Dict[str, BaseSkill] = {}
        self._aliases: Dict[str, str] = {}
        self.cortex = cortex
        self.completer = completer

        # ---------------------------------------------------------------------
        # 1. Register Agentic Cognitive Skills
        # ---------------------------------------------------------------------
        self.register(
            PlanningSkill(),
            aliases=["plan", "planning", "5w1h", "dag", "blueprint"],
        )
        self.register(
            AskingSkill(),
            aliases=["ask", "asking", "clarify", "clarification", "question"],
        )
        self.register(
            SuggestingSkill(),
            aliases=["suggest", "suggesting", "recommend", "recommendation", "propose"],
        )
        self.register(
            GrillingSkill(),
            aliases=["grill", "grilling", "interview", "challenge", "stress_test"],
        )
        self.register(
            SchedulingSkill(cortex=cortex),
            aliases=["schedule", "scheduling", "timer", "cron"],
        )
        self.register(
            GoalSkill(completer=completer),
            aliases=["goal", "autonomous_goal", "solve", "achieve"],
        )
        self.register(
            AlignerSkill(),
            aliases=["align", "aligner", "reconcile", "drift"],
        )

        # ---------------------------------------------------------------------
        # 2. Register Software Engineering Skills
        # ---------------------------------------------------------------------
        self.register(
            SecurityAuditSkill(),
            aliases=["security", "sec", "audit", "vuln"],
        )
        self.register(
            TestGenerationSkill(),
            aliases=["test", "tests", "unit_test", "pytest"],
        )
        self.register(
            RefactoringSkill(),
            aliases=["refactor", "modernize", "clean_code"],
        )
        self.register(
            ComplexityAnalysisSkill(),
            aliases=["complexity", "big_o", "perf", "bigo"],
        )
        self.register(
            ApiScaffoldingSkill(),
            aliases=["api", "scaffold_api", "fastapi", "router"],
        )
        self.register(
            DocumentationSkill(),
            aliases=["docs", "generate_docs", "docstring", "spec"],
        )
        self.register(
            SqlOptimizationSkill(),
            aliases=["sql", "optimize_sql", "db_opt"],
        )
        self.register(
            SynapticMemorySkill(cortex=cortex),
            aliases=["synapse", "memory", "synaptic_memory"],
        )

    def set_completer(self, completer: Any) -> None:
        """Inject task completer into goal and cognitive skills."""
        self.completer = completer
        goal_skill = self._skills.get("goal")
        if goal_skill and hasattr(goal_skill, "set_completer"):
            goal_skill.set_completer(completer)

    def set_cortex(self, cortex: Any) -> None:
        """Inject cortex instance into scheduling and synaptic skills."""
        self.cortex = cortex
        sched_skill = self._skills.get("scheduling")
        if sched_skill and hasattr(sched_skill, "set_cortex"):
            sched_skill.set_cortex(cortex)
        mem_skill = self._skills.get("synaptic_memory")
        if mem_skill and hasattr(mem_skill, "set_cortex"):
            mem_skill.set_cortex(cortex)

    def register(self, skill: BaseSkill, aliases: Optional[List[str]] = None) -> None:
        """Register a new skill with optional shorthand aliases."""
        self._skills[skill.name] = skill
        if aliases:
            for alias in aliases:
                self._aliases[alias.lower()] = skill.name

    def get(self, name_or_alias: str) -> Optional[BaseSkill]:
        """Lookup skill by exact name or registered alias."""
        clean = name_or_alias.strip().lower()
        if clean in self._skills:
            return self._skills[clean]
        canonical = self._aliases.get(clean)
        if canonical and canonical in self._skills:
            return self._skills[canonical]
        # Fuzzy match
        matches = difflib.get_close_matches(clean, list(self._skills.keys()) + list(self._aliases.keys()), n=1, cutoff=0.6)
        if matches:
            target = matches[0]
            canonical = self._aliases.get(target, target)
            return self._skills.get(canonical)
        return None

    def execute(self, skill_name_or_alias: str, target: str, context: Optional[Dict[str, Any]] = None) -> SkillResult:
        """Execute a skill by name or alias."""
        skill = self.get(skill_name_or_alias)
        if not skill:
            available = ", ".join(sorted(self._skills.keys()))
            return SkillResult(
                skill_name=skill_name_or_alias,
                success=False,
                output=f"Skill '{skill_name_or_alias}' not found. Available skills: {available}",
                execution_time_ms=0.0,
            )
        return skill.execute(target=target, context=context)

    def list_skills(self) -> List[Dict[str, Any]]:
        """Return structured summary of all registered skills."""
        res: List[Dict[str, Any]] = []
        for name, skill in sorted(self._skills.items()):
            aliases = [a for a, canonical in self._aliases.items() if canonical == name]
            res.append({
                "name": name,
                "category": skill.category,
                "description": skill.description,
                "aliases": aliases,
            })
        return res

    def list_cognitive_skills(self) -> List[Dict[str, Any]]:
        """Return only agentic cognitive skills."""
        cognitive_cats = {"planning", "asking", "suggesting", "grilling", "scheduling", "goal", "alignment"}
        return [s for s in self.list_skills() if s["category"] in cognitive_cats]

    def list_engineering_skills(self) -> List[Dict[str, Any]]:
        """Return software engineering and analysis skills."""
        cognitive_cats = {"planning", "asking", "suggesting", "grilling", "scheduling", "goal", "alignment"}
        return [s for s in self.list_skills() if s["category"] not in cognitive_cats]

    def __iter__(self):
        return iter(self._skills.values())

    def __len__(self) -> int:
        return len(self._skills)

    def __repr__(self) -> str:
        return f"<SkillRegistry [{len(self._skills)} skills registered]>"
