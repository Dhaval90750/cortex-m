"""Metacognitive Reasoning & Proactive Suggestion Engine for CortexLM.

Implements:
1. 5W1H Metacognitive Reasoning Framework:
   - WHEN to think fast (O(1) direct recall) vs deliberative deep reflection.
   - WHY to respond authoritatively vs seek epistemic clarification.
   - WHAT concepts and proactive suggestions to offer.
   - WHERE knowledge is grounded (synaptic attractor, source provenance, resonance).
   - HOW to adapt comprehension level (ELI5, Executive, Senior Engineer, JSON).

2. Proactive Active Learning Engine:
   - Generates thoughtful, high-value suggestions based on domain & context.
   - Learns user patterns and architectural preferences live into synaptic weights.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from cortex_m.model.cortex_lm import CortexLM


@dataclass
class MetacognitiveAssessment:
    when_mode: str  # "FAST_O1_PATHWAY" | "DELIBERATIVE_PREFRONTAL"
    why_rationale: str
    what_topic: str
    where_attractor: str
    how_comprehension_level: str  # "ELI5_ANALOGY" | "EXECUTIVE_TLDR" | "SENIOR_ENGINEER" | "STANDARD_EXPERT"
    suggested_followups: List[str] = field(default_factory=list)
    active_learning_prompt: Optional[str] = None


class MetacognitiveEngine:
    """Orchestrates 5W1H metacognitive reasoning and proactive suggestions."""

    def __init__(self, lm: CortexLM):
        self.lm = lm
        self.user_learned_patterns: Dict[str, str] = {}

    def assess_query(
        self,
        prompt: str,
        attractor_id: str,
        hallucination_risk: float,
        is_grounded: bool,
    ) -> MetacognitiveAssessment:
        """Evaluate 5W1H metacognition for prompt and determine suggestions."""
        p_lower = prompt.lower()

        # 1. WHEN: Fast O(1) vs Deliberative
        fast_triggers = ["hello", "hi", "hey", "what is your name", "who are you", "what is 2+", "ping"]
        if any(trig in p_lower for trig in fast_triggers) or len(p_lower.split()) <= 3:
            when_mode = "FAST_O1_PATHWAY"
        else:
            when_mode = "DELIBERATIVE_PREFRONTAL"

        # 2. HOW: Comprehension Level
        if any(kw in p_lower for kw in ["eli5", "explain simply", "like i'm 5", "for a beginner", "layman terms", "simple analogy"]):
            how_level = "ELI5_ANALOGY"
        elif any(kw in p_lower for kw in ["executive", "tldr", "in short", "high level", "bottom line"]):
            how_level = "EXECUTIVE_TLDR"
        elif any(kw in p_lower for kw in ["deep dive", "internals", "under the hood", "production grade", "advanced", "kernel"]):
            how_level = "SENIOR_ENGINEER"
        else:
            how_level = "STANDARD_EXPERT"

        # 3. WHY: Epistemic rationale
        if is_grounded and hallucination_risk < 0.25:
            why_rationale = "High epistemic confidence: cue settled into deep, stable Lyapunov attractor basin."
        elif is_grounded:
            why_rationale = "Moderate confidence: verified attractor match with positive pathway resonance."
        else:
            why_rationale = "Epistemic uncertainty: cue did not settle in stored basins; abstaining from confabulation."

        # 4. WHAT: Topic & Proactive Suggestions
        topic = attractor_id.replace("_", " ").title()
        suggestions: List[str] = []
        active_learning_q: Optional[str] = None

        if "db" in attractor_id or "sql" in p_lower or "database" in p_lower or "index" in p_lower:
            suggestions.append("Would you like me to inspect your query's WHERE clause for optimal B-Tree index coverage?")
            suggestions.append("Should we review EXPLAIN ANALYZE execution cost or connection pooling setup?")
            active_learning_q = "What database engine and version are you running? (Tell me so I remember your stack)"

        elif "docker" in p_lower or "vm" in p_lower or "container" in p_lower:
            suggestions.append("Would you like a production multi-stage Dockerfile template with minimal image footprint?")
            suggestions.append("Should we analyze cgroups memory limit configurations to prevent OOM kills?")
            active_learning_q = "Are you deploying on Kubernetes, Docker Swarm, or bare VMs?"

        elif "rebase" in p_lower or "git" in p_lower or "merge" in p_lower:
            suggestions.append("Would you like to review how to safely resolve rebase conflicts without losing commits?")
            suggestions.append("Should we set up a clean git commit hygiene workflow for your team?")

        elif "leak" in p_lower or "memory" in p_lower:
            suggestions.append("Would you like a walkthrough on capturing heap dumps using runtime profilers?")
            suggestions.append("Should we check for circular reference memory leaks or unclosed socket descriptors?")

        elif "rust" in p_lower:
            suggestions.append("Would you like to explore how Rust's borrow checker prevents data races at compile time?")
            suggestions.append("Should we look at zero-cost abstractions and lifetime annotations?")

        elif "cortex" in p_lower or "hopfield" in p_lower or "neural" in p_lower:
            suggestions.append("Would you like to inspect continuous Lyapunov energy descent equations?")
            suggestions.append("Should we run a live anti-Hebbian unlearning demonstration in < 1 ms?")

        return MetacognitiveAssessment(
            when_mode=when_mode,
            why_rationale=why_rationale,
            what_topic=topic,
            where_attractor=attractor_id,
            how_comprehension_level=how_level,
            suggested_followups=suggestions,
            active_learning_prompt=active_learning_q,
        )

    def process_user_feedback(self, feedback_text: str) -> Optional[Dict[str, Any]]:
        """Active learning: detect user preferences and learn directly into synapses."""
        f_lower = feedback_text.lower()

        # Database stack learning
        db_match = re.search(r"\b(postgres|postgresql|mysql|sqlite|mongodb|redis|cassandra|rocksdb)\b", f_lower)
        if db_match and any(kw in f_lower for kw in ["we use", "i use", "our db", "our database", "running", "stack"]):
            db_engine = db_match.group(1).title()
            mem_id = f"user_db_stack_{db_engine.lower()}"
            knowledge = f"The user project runs {db_engine} as their primary database engine in production."
            res = self.lm.learn(knowledge, memory_id=mem_id, source="user_active_learning_feedback")
            self.user_learned_patterns["database"] = db_engine
            return {
                "pattern_type": "database_stack",
                "value": db_engine,
                "memory_id": mem_id,
                "status": "learned_into_synapses",
            }

        # Language stack learning
        lang_match = re.search(r"\b(python|rust|golang|go|typescript|javascript|c\+\+|java)\b", f_lower)
        if lang_match and any(kw in f_lower for kw in ["we write", "i code in", "in our codebase", "primary language"]):
            lang = lang_match.group(1).title()
            mem_id = f"user_lang_stack_{lang.lower()}"
            knowledge = f"The user codebase is built primarily in {lang}."
            res = self.lm.learn(knowledge, memory_id=mem_id, source="user_active_learning_feedback")
            self.user_learned_patterns["language"] = lang
            return {
                "pattern_type": "language_stack",
                "value": lang,
                "memory_id": mem_id,
                "status": "learned_into_synapses",
            }

        return None
