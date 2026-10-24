"""Lightweight Intent Router & Dynamic Agent Gatekeeper for Cortex-M.

Solves the 'hammer and nail' problem by classifying user input upstream:
1. CONCEPTUAL (Q&A, explanations, theories, analogies) -> Bypasses swarm, returns clean Markdown.
2. ENGINEERING (Code implementation, bug fixes, refactoring) -> Triggers multi-agent swarm.
3. CONVERSATIONAL (Social greetings, check-ins, identity, user name) -> Fast human resonance.
4. SKILL (Explicit /command or registered capability) -> Routes to registered skill.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class UserIntent(enum.Enum):
    CONCEPTUAL = "conceptual"
    ENGINEERING = "engineering"
    CONVERSATIONAL = "conversational"
    SKILL = "skill"


@dataclass
class IntentClassification:
    intent: UserIntent
    confidence: float
    reason: str
    target: str = ""
    skill_name: Optional[str] = None


class IntentRouter:
    """Fast, deterministic upstream classifier gatekeeper for the CLI."""

    # Conceptual / Explanatory Inquiry Patterns
    CONCEPTUAL_PATTERNS = [
        r"^what\s+is\b",
        r"^what\s+are\b",
        r"^how\s+does\b",
        r"^how\s+do\b",
        r"^why\s+is\b",
        r"^why\s+does\b",
        r"^why\s+do\b",
        r"^explain\b",
        r"^tell\s+me\s+about\b",
        r"^i\s+want\s+to\s+understand\b",
        r"^i\s+want\s+to\s+know\b",
        r"^can\s+you\s+explain\b",
        r"^teach\s+me\b",
        r"^help\s+me\s+understand\b",
        r"^difference\s+between\b",
        r"^compare\b",
        r"^what's\s+the\s+difference\b",
        r"^overview\s+of\b",
        r"^meaning\s+of\b",
        r"^definition\s+of\b",
        r"^walk\s+me\s+through\b",
        r"^pros\s+and\s+cons\b",
    ]

    # Explicit Actionable Engineering / Coding Verbs
    ENGINEERING_VERBS = [
        "implement", "build", "create", "write code", "develop", "scaffold",
        "refactor", "debug", "fix", "generate", "code a", "program a",
        "make an api", "write a script", "write a class", "write a function",
        "write tests", "test suite", "dockerize", "optimize query", "benchmark"
    ]

    # Conversational Social Patterns
    CONVERSATIONAL_EXACT = {
        "hi", "hello", "hey", "how are you", "who are you", "what can you do",
        "help", "bye", "goodbye", "thanks", "thank you", "my name is", "cortex"
    }

    def classify(self, user_input: str) -> IntentClassification:
        """Categorize prompt into CONCEPTUAL, ENGINEERING, CONVERSATIONAL, or SKILL."""
        raw = user_input.strip()
        cleaned = raw.lower()

        if not cleaned:
            return IntentClassification(
                intent=UserIntent.CONVERSATIONAL,
                confidence=1.0,
                reason="Empty input",
            )

        # 1. Explicit Skill Directives (e.g. /plan, /sec, /api, /goal)
        if raw.startswith("/"):
            parts = raw[1:].split(maxsplit=1)
            skill = parts[0].lower()
            target = parts[1].strip() if len(parts) > 1 else ""
            return IntentClassification(
                intent=UserIntent.SKILL,
                confidence=1.0,
                reason=f"Explicit skill directive: /{skill}",
                target=target,
                skill_name=skill,
            )

        # 2. Conversational Social Interventions
        if cleaned in self.CONVERSATIONAL_EXACT or any(cleaned.startswith(f"{p} ") for p in ["my name is", "i am", "call me"]):
            return IntentClassification(
                intent=UserIntent.CONVERSATIONAL,
                confidence=0.95,
                reason="Social conversational greeting or profile disclosure",
                target=raw,
            )

        # 3. Check for Conceptual / Q&A Patterns First
        for pattern in self.CONCEPTUAL_PATTERNS:
            if re.search(pattern, cleaned):
                # Ensure the user isn't actually asking to write code (e.g., "explain how to implement an lru cache with code")
                is_explicit_code = any(kw in cleaned for kw in ["write code", "implement in python", "give me code", "generate code"])
                if not is_explicit_code:
                    return IntentClassification(
                        intent=UserIntent.CONCEPTUAL,
                        confidence=0.95,
                        reason=f"Matched conceptual explanation pattern: {pattern}",
                        target=raw,
                    )

        # 4. Questions with question marks without implementation verbs
        if cleaned.endswith("?") and not any(v in cleaned for v in self.ENGINEERING_VERBS):
            return IntentClassification(
                intent=UserIntent.CONCEPTUAL,
                confidence=0.90,
                reason="Inquiry question without implementation command",
                target=raw,
            )

        # 5. Engineering Verbs -> Swarm Execution
        if any(v in cleaned for v in self.ENGINEERING_VERBS):
            return IntentClassification(
                intent=UserIntent.ENGINEERING,
                confidence=0.95,
                reason="Explicit software engineering action verb detected",
                target=raw,
            )

        # 6. Default Fallback
        # If words < 4 and abstract, treat as conceptual question; if longer instructions, treat as engineering goal
        word_count = len(cleaned.split())
        if word_count <= 4 and not any(w in cleaned for w in ["service", "cache", "api", "function", "class", "algorithm"]):
            return IntentClassification(
                intent=UserIntent.CONCEPTUAL,
                confidence=0.75,
                reason="Short abstract conceptual inquiry",
                target=raw,
            )

        return IntentClassification(
            intent=UserIntent.ENGINEERING,
            confidence=0.85,
            reason="Actionable engineering task description",
            target=raw,
        )
