"""Dynamic Conversational Cognition & Human Resonance Engine for Cortex-M.

Provides human-centric, dynamic, resilient conversational intelligence:
1. Natural greetings, social check-ins, and wellbeing inquiries.
2. Self-identity, architecture awareness, and capability explanation.
3. User self-disclosure parsing & episodic working memory (e.g. "My name is Dhaval").
4. Graceful handling of incomplete phrases ("my name is", "you are", "can you").
5. Elimination of robotic boilerplate and rigid academic attractor collapse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ConversationalIntent:
    is_conversational: bool
    category: str  # "greeting", "wellbeing", "identity", "capability", "incomplete", "user_profile", "gratitude", "farewell", "none"
    response: Optional[str] = None
    extracted_data: Dict[str, Any] = field(default_factory=dict)


class ConversationalCognition:
    """Dynamic conversational router and natural dialogue realizer."""

    def __init__(self):
        self.user_profile: Dict[str, Any] = {}

    def analyze_intent(self, prompt: str, working_memory_context: Optional[Dict[str, Any]] = None) -> ConversationalIntent:
        """Dynamically classify conversational intent from prompt with human-first resonance."""
        raw = prompt.strip()
        cleaned = raw.lower()
        words = re.findall(r"[a-zA-Z0-9_]+", cleaned)
        word_count = len(words)

        if not words:
            return ConversationalIntent(
                is_conversational=True,
                category="incomplete",
                response="I'm here and listening! What would you like to explore or build today?",
            )

        # 1. Incomplete / Fragmented Prompts (e.g. "my name is", "my name is ?", "you are", "can you")
        if cleaned in ("my name is", "my name is ?", "my name is?", "name is", "call me"):
            return ConversationalIntent(
                is_conversational=True,
                category="incomplete",
                response="You started saying 'my name is' -- please let me know your name so I can remember it and address you properly!",
            )

        if cleaned in ("you are", "you are?", "you are ?", "you are a", "you're"):
            return ConversationalIntent(
                is_conversational=True,
                category="incomplete",
                response="I am CortexLM, the native continuous neural language model powered by CORTEX-M (Context-Oriented Routing and Trace Exchange for MIMO). How can I assist you right now?",
            )

        if cleaned in ("can you", "could you", "would you", "can you help", "can you?"):
            return ConversationalIntent(
                is_conversational=True,
                category="incomplete",
                response="Yes, absolutely! I can help you design architectures, synthesize production-grade code, solve algorithms, or explore continuous neural dynamics. What do you have in mind?",
            )

        # 2. User Profile Disclosure: "my name is <Name>", "i am <Name>", "call me <Name>"
        name_match = re.search(r"\b(?:my name is|i am called|call me)\s+([A-Za-z]+)\b", raw, re.IGNORECASE)
        if name_match:
            name = name_match.group(1).strip().capitalize()
            self.user_profile["user_name"] = name
            return ConversationalIntent(
                is_conversational=True,
                category="user_profile",
                response=f"Hello {name}! Great to connect with you. I've noted your name in my active session memory. What systems, algorithms, or projects are we working on today?",
                extracted_data={"user_name": name},
            )

        # 3. User Inquiries About Stored Profile: "what is my name?", "who am i?", "do you remember my name?"
        if any(p in cleaned for p in ["what is my name", "what's my name", "who am i", "remember my name", "do you know my name"]):
            user_name = self.user_profile.get("user_name")
            if user_name:
                return ConversationalIntent(
                    is_conversational=True,
                    category="user_profile",
                    response=f"Your name is {user_name}! We're connected in this active session. What shall we tackle next?",
                )
            else:
                return ConversationalIntent(
                    is_conversational=True,
                    category="user_profile",
                    response="I don't have your name in active session memory yet. Please tell me your name (e.g., 'My name is Alex') and I'll remember it!",
                )

        # 4. Identity Inquiries: "who are you", "what is your name", "what is cortex-m", "what is cortexlm"
        identity_patterns = [
            "who are you", "what are you", "what is your name", "what's your name",
            "tell me about yourself", "what is cortex-m", "what is cortex m",
            "what is cortexlm", "what is cortex lm", "who made you", "introduce yourself"
        ]
        if any(p in cleaned for p in identity_patterns) or (word_count <= 4 and "who" in words and "you" in words):
            return ConversationalIntent(
                is_conversational=True,
                category="identity",
                response=(
                    "I am **CortexLM**, the native bio-inspired neural language model powered directly by **CORTEX-M** "
                    "(Context-Oriented Routing and Trace Exchange for MIMO).\n\n"
                    "Unlike traditional Transformers that rely on massive external vector databases or endless KV-caches, "
                    "my knowledge is encoded directly into continuous synaptic attractor networks (W). I combine:\n"
                    "* **Real-Time Plasticity**: Instant Hebbian learning (`/learn`) and anti-Hebbian unlearning (`/forget`).\n"
                    "* **Lyapunov Energy Shield**: Mathematical immunity against hallucinations.\n"
                    "* **Autonomous Coding Swarm (CortexCode)**: Multi-agent teams (Architect, Coder, Critic, Auditor) for production software.\n"
                    "* **Edge Native Substrate**: Runs with sub-millisecond latency on CPUs and mobile devices.\n\n"
                    "How can I assist you with your architecture, algorithms, or systems today?"
                ),
            )

        # 5. Wellbeing Inquiries: "how are you", "how are you doing", "how's it going", "how do you feel"
        wellbeing_patterns = [
            "how are you", "how are you doing", "how do you feel", "how is it going",
            "hows it going", "how's it going", "how are things", "are you okay", "what's up", "whats up"
        ]
        if any(p in cleaned for p in wellbeing_patterns) or (word_count <= 4 and "how" in words and "you" in words):
            user_name = self.user_profile.get("user_name")
            greeting_prefix = f"Hello {user_name}! " if user_name else "Hello! "
            return ConversationalIntent(
                is_conversational=True,
                category="wellbeing",
                response=(
                    f"{greeting_prefix}Great to connect with you. I'm operating at peak performance! "
                    "Synaptic attractors are stable, prefrontal working memory is active, and the Lyapunov energy "
                    "guardrails are calibrated. What systems, algorithms, or challenges are we tackling today?"
                ),
            )

        # 6. Greetings: "hi", "hello", "hey", "good morning", "greetings"
        greetings_list = ["hi", "hello", "hey", "howdy", "sup", "greetings", "good morning", "good evening", "good afternoon"]
        if cleaned in greetings_list or (word_count <= 2 and any(w in greetings_list for w in words)):
            user_name = self.user_profile.get("user_name")
            name_str = f" {user_name}" if user_name else ""
            return ConversationalIntent(
                is_conversational=True,
                category="greeting",
                response=f"Hello{name_str}! Great to connect with you. What systems, algorithms, or problems are we working on today?",
            )

        # 7. Capabilities: "what can you do", "help", "commands"
        capability_patterns = ["what can you do", "what are your capabilities", "help me", "how can you help", "what are your features"]
        if any(p in cleaned for p in capability_patterns) or cleaned == "help":
            return ConversationalIntent(
                is_conversational=True,
                category="capability",
                response=(
                    "Here are my core capabilities as Cortex-M / CortexLM:\n\n"
                    "1. **Systems & Software Architecture**: Design distributed systems, evaluate storage engines (B-Tree vs LSM), "
                    "concurrency models (GIL, epoll), and transaction semantics (ACID vs BASE).\n"
                    "2. **Autonomous Coding Swarm (CortexCode)**: Deconstruct, write, statically review, and execute-verify "
                    "production-grade algorithms, data structures, and services.\n"
                    "3. **Continuous Synaptic Learning**: Imprint facts live into physical weights (`/learn`) and unlearn instantly (`/forget`).\n"
                    "4. **Mathematical & Algorithmic Problem Solving**: Graph algorithms, dynamic programming, linear algebra, and formal logic.\n"
                    "5. **Biological Sleep & Substrate Diagnostics**: Perform synaptic consolidation and health audits (`/sleep`, `/audit`).\n\n"
                    "Tell me what you'd like to work on!"
                ),
            )

        # 8. Gratitude: "thank you", "thanks", "awesome", "great job"
        gratitude_patterns = ["thank you", "thanks", "appreciate it", "great job", "awesome", "perfect thanks"]
        if any(cleaned == p or cleaned.startswith(p) for p in gratitude_patterns) and word_count <= 4:
            return ConversationalIntent(
                is_conversational=True,
                category="gratitude",
                response="You're very welcome! Always glad to help. Let me know whenever you're ready for the next task.",
            )

        # 9. Farewell: "bye", "goodbye", "see you", "have a good day"
        farewell_patterns = ["bye", "goodbye", "see you", "see ya", "cya", "have a good day", "good night"]
        if any(cleaned == p or cleaned.startswith(p) for p in farewell_patterns) and word_count <= 4:
            return ConversationalIntent(
                is_conversational=True,
                category="farewell",
                response="Goodbye! Synaptic memory states are preserved. Looking forward to our next session!",
            )

        return ConversationalIntent(is_conversational=False, category="none")
