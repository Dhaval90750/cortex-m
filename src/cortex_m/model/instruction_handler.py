"""Instruction Awareness & Conversational Realization Engine for CortexLM.

Transforms raw energy-recalled facts into natural, human-grade, instruction-compliant
responses without robotic boilerplate or canned AI disclaimers.

Supports:
1. Directive Parsing:
   - Formats: JSON, Markdown tables, Bullet lists, Code-only, Plain text.
   - Constraints: 'in 1 sentence', 'step-by-step', 'concise / tldr', 'detailed / in-depth'.
   - Style: Natural human tone (direct, articulate, no 'As an AI language model...').
2. Syntactic & Numerical Preservation:
   - Injects verbatim numbers, code snippets, and variable names from Dorsal Stream.
3. Clean Provenance Badging:
   - Exposes causal provenance metadata cleanly without cluttering natural dialog.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


class InstructionHandler:
    """Parses instructions from user prompts and shapes grounded model responses."""

    def __init__(self):
        # Patterns for detecting formatting directives
        self.json_patterns = re.compile(r"\b(json|as json|format as json|return json)\b", re.IGNORECASE)
        self.bullet_patterns = re.compile(r"\b(bullet points?|as bullets?|list format|key points)\b", re.IGNORECASE)
        self.table_patterns = re.compile(r"\b(as a table|markdown table|table format)\b", re.IGNORECASE)
        self.code_patterns = re.compile(r"\b(code only|write code|code snippet|in python|in rust|implement)\b", re.IGNORECASE)
        self.step_patterns = re.compile(r"\b(step by step|steps|how to|guide|walkthrough)\b", re.IGNORECASE)
        self.concise_patterns = re.compile(r"\b(in 1 sentence|one sentence|concise|briefly|tldr|short)\b", re.IGNORECASE)
        self.in_depth_patterns = re.compile(r"\b(in depth|deep dive|comprehensive|explain thoroughly)\b", re.IGNORECASE)

    def parse_instruction(self, prompt: str) -> Dict[str, Any]:
        """Extract explicit and implicit instructions from the prompt."""
        p_clean = prompt.strip()
        lower = p_clean.lower()

        wants_json = bool(self.json_patterns.search(lower))
        wants_bullets = bool(self.bullet_patterns.search(lower))
        wants_table = bool(self.table_patterns.search(lower))
        wants_code = bool(self.code_patterns.search(lower))
        wants_steps = bool(self.step_patterns.search(lower))
        wants_concise = bool(self.concise_patterns.search(lower))
        wants_in_depth = bool(self.in_depth_patterns.search(lower))

        # Detect target language if code requested
        code_lang = "python"
        if "rust" in lower:
            code_lang = "rust"
        elif "bash" in lower or "shell" in lower:
            code_lang = "bash"
        elif "sql" in lower:
            code_lang = "sql"
        elif "javascript" in lower or "js" in lower:
            code_lang = "javascript"

        return {
            "wants_json": wants_json,
            "wants_bullets": wants_bullets,
            "wants_table": wants_table,
            "wants_code": wants_code,
            "wants_steps": wants_steps,
            "wants_concise": wants_concise,
            "wants_in_depth": wants_in_depth,
            "code_lang": code_lang,
        }

    def format_realization(
        self,
        prompt: str,
        fact_text: str,
        memory_id: str,
        source: str,
        resonance_pct: int,
        dorsal_meta: Optional[Dict[str, Any]] = None,
        include_badge: bool = False,
    ) -> str:
        """Format grounded knowledge to strictly adhere to instructions in prompt."""
        spec = self.parse_instruction(prompt)
        d_meta = dorsal_meta or {}
        numbers = d_meta.get("numbers_detected", [])

        # Remove canned / robotic phrases if present
        clean_fact = fact_text.strip()
        for robotic in [
            "As an AI language model,",
            "As an AI,",
            "I am programmed to",
            "Certainly!",
            "Sure! Here is the answer:",
        ]:
            if clean_fact.startswith(robotic):
                clean_fact = clean_fact[len(robotic) :].strip()

        # 1. Format as JSON if user requested
        if spec["wants_json"]:
            payload = {
                "topic": memory_id.replace("_", " ").title(),
                "summary": clean_fact,
                "verified": True,
                "provenance": {
                    "source": source,
                    "attractor": memory_id,
                    "resonance_pct": resonance_pct,
                },
            }
            if numbers:
                payload["operands"] = numbers
            return json.dumps(payload, indent=2)

        # 2. Adaptive Comprehension: ELI5 / Plain Human Analogies
        wants_eli5 = any(kw in prompt.lower() for kw in [
            "eli5", "explain simply", "like i'm 5", "like im 5", "for a beginner",
            "layman", "simple analogy", "simple terms", "simple english", "plain english"
        ])
        if wants_eli5:
            analogy = self._get_intuitive_analogy(memory_id, clean_fact)
            if include_badge:
                return f"{analogy} (Remembered from {source}, Pathway Resonance: {resonance_pct}%)"
            return analogy

        # 3. Format as Bullet Points if requested
        if spec["wants_bullets"]:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_fact) if s.strip()]
            bullets = [f"• {s}" for s in sentences]
            res = "\n".join(bullets)
            if include_badge:
                res += f"\n\n*(Source: {source}, Resonance: {resonance_pct}%)*"
            return res

        # 4. Format as Step-by-Step if requested
        if spec["wants_steps"]:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_fact) if s.strip()]
            steps = [f"{i+1}. {s}" for i, s in enumerate(sentences)]
            res = "\n".join(steps)
            if include_badge:
                res += f"\n\n*(Source: {source}, Resonance: {resonance_pct}%)*"
            return res

        # 5. Format as Code if requested
        if spec["wants_code"] and "```" not in clean_fact:
            lang = spec["code_lang"]
            if "def " in clean_fact or "fn " in clean_fact or "{" in clean_fact:
                res = f"```{lang}\n{clean_fact}\n```"
            else:
                res = f"{clean_fact}\n\n```{lang}\n# Implementation details for {memory_id}\n```"
            if include_badge:
                res += f"\n*(Source: {source}, Resonance: {resonance_pct}%)*"
            return res

        # 6. Format as Concise / One-Sentence if requested
        if spec["wants_concise"]:
            match = re.match(r"^([^.!?]+[.!?])", clean_fact)
            res = match.group(1).strip() if match else clean_fact
            if include_badge:
                res += f" ({source}, Resonance: {resonance_pct}%)"
            return res

        # 7. Natural Human Conversational Realization
        if memory_id in ("conversational_greeting", "conversational_help"):
            return clean_fact

        # Standard natural expert response
        if include_badge:
            return f"{clean_fact} (Remembered from {source}, Pathway Resonance: {resonance_pct}%)"
        return clean_fact

    def _get_intuitive_analogy(self, memory_id: str, default_fact: str) -> str:
        """Translate dense technical knowledge into vivid, plain-English analogies."""
        analogies = {
            "docker_vs_vms": (
                "Think of Docker like renting a room in a shared apartment: you share the building's plumbing "
                "and foundation (the OS kernel), which makes moving in super fast and lightweight. Virtual Machines, "
                "on the other hand, are like building a completely separate standalone house with its own private "
                "foundation and utilities."
            ),
            "git_merge_vs_rebase": (
                "Think of git merge like gluing two tree branches together where they touch, leaving a visible knot "
                "where they met. Git rebase is like gently lifting your branch off and replanting it right onto the "
                "very top of the main trunk, making the whole history look like a single clean straight line."
            ),
            "memory_leak_debugging": (
                "Think of a memory leak like saving delivery boxes in your attic and forgetting to throw them away. "
                "At first everything seems fine, but eventually the attic is stuffed and nothing else fits. "
                "Debugging it means taking an inventory snapshot to see which boxes you no longer need."
            ),
            "rust_memory_safety": (
                "Think of Rust's borrow checker like a strict librarian: multiple people can read the same book at once, "
                "or one person can take notes in it alone, but nobody is allowed to write in the book while others are reading it."
            ),
            "sql_query_optimization": (
                "Think of SQL query optimization like a GPS planning the fastest driving route: instead of driving through "
                "every neighborhood street (sequential scan), it checks highway traffic and takes the direct expressway (index scan)."
            ),
            "btree_vs_lsm_storage": (
                "Think of a B-Tree like a neatly organized library bookshelf where every book has its exact sorted spot (great for quick lookups). "
                "An LSM-Tree is like quickly scribbling notes in an inbox journal on your desk and only sorting them onto the shelves later in batches (great for super fast writing)."
            ),
            "acid_vs_base_transactions": (
                "Think of ACID like an ATM cash withdrawal: either you get all your money and your bank balance updates together, "
                "or the whole transaction cancels—no middle ground. BASE is like sending a package in the mail: it will eventually "
                "arrive and settle, but the status updates along the way."
            ),
            "bayes_theorem": (
                "Think of Bayes' Theorem like updating your weather forecast: if you hear thunder outside (new evidence), "
                "you immediately update your morning belief from 'it might rain' to 'it is almost certainly raining'."
            ),
            "speed_of_light_constant": (
                "The speed of light is the universe's ultimate cosmic speed limit: absolutely nothing can travel faster than it, "
                "just like how you cannot drive faster than physical reality allows."
            ),
            "first_principles_thinking": (
                "First-principles thinking is like dismantling a LEGO tower down to the individual building blocks to see how "
                "each piece actually fits together, instead of just copying someone else's finished building."
            ),
            "pareto_principle_optimization": (
                "The Pareto Principle (the 80/20 rule) means that in most software systems, 80% of your performance bottlenecks "
                "come from just 20% of the code. Focus on fixing that critical 20% to get massive speedups."
            ),
        }
        return analogies.get(memory_id, f"In simple terms: {default_fact}")
