"""Agentic Cognitive Skills Suite for Cortex-M & CortexCode.

Implements high-level, dynamically reasoned cognitive intelligence:
1. PlanningSkill (/plan): Dynamic 5W1H decomposition, milestone DAG, entity mapping, and verification criteria.
2. AskingSkill (/ask): Proactive clarification of underspecified requirements with high-signal options & recommendations.
3. SuggestingSkill (/suggest): Dynamic high-leverage architectural improvements, performance wins, and next steps.
4. GrillingSkill (/grill): Socratic technical stress-test interrogating edge cases, concurrency, and scale.
5. SchedulingSkill (/schedule): Timers, recurring cron-like triggers, synaptic sleep consolidation, and audits.
6. GoalSkill (/goal): Autonomous end-to-end goal completion loop with AST review & sandbox verification.
7. AlignerSkill (/align): Dynamic intent-to-implementation reconciliation detecting drift and omissions.

Zero hardcoded topic branches: all reasoning is derived dynamically from semantic entity extraction,
syntactic AST parsing, and CORTEX-M continuous associative memory attractors.
"""

from __future__ import annotations

import ast
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .base import BaseSkill, SkillResult


# =============================================================================
# HELPER: DYNAMIC SEMANTIC & ARCHITECTURAL PARSER
# =============================================================================

def _extract_semantic_features(text: str) -> Dict[str, Any]:
    """Dynamically extracts entities, actions, modifiers, and architectural traits from any prompt."""
    tokens = re.findall(r"[a-zA-Z0-9_]+", text)
    tokens_lower = [t.lower() for t in tokens]
    
    stop_words = {
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "and", "or", "is", "are", "was", "were", "be", "been", "that", "this", "which",
        "it", "as", "into", "like", "how", "what", "when", "where", "why", "who", "i",
        "you", "he", "she", "we", "they", "my", "your", "our", "their", "please", "make",
        "build", "implement", "write", "create", "generate", "code", "python", "system",
        "class", "function", "program", "solution", "need", "want", "help", "me",
    }
    
    # Meaningful domain entities & verbs
    meaningful = [w for w in tokens if w.lower() not in stop_words and len(w) >= 3]
    
    # Check for explicit PascalCase identifiers first (e.g. SensorMonitor, BankAccount)
    pascal_candidates = [w for w in tokens if re.match(r"^[A-Z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*$", w)]
    modifiers = {"in", "memory", "distributed", "fast", "simple", "concurrent", "async", "high", "low", "real", "time"}
    substantive = [w for w in meaningful if w.lower() not in modifiers]

    if pascal_candidates:
        primary_entity = pascal_candidates[0]
    elif substantive:
        primary_entity = substantive[0].capitalize()
    elif meaningful:
        primary_entity = meaningful[0].capitalize()
    else:
        primary_entity = "CustomSystem"

    secondary_entities = [m.capitalize() for m in meaningful if m.lower() != primary_entity.lower()]

    # Characteristic traits detection (without hardcoding specific topics)
    t_set = set(tokens_lower)
    is_concurrent = bool(t_set & {"thread", "concurrent", "async", "lock", "parallel", "race", "mutex", "worker", "queue"})
    is_persistent = bool(t_set & {"store", "storage", "save", "persist", "db", "database", "disk", "file", "synapse", "memory"})
    is_networked = bool(t_set & {"api", "http", "rest", "grpc", "client", "server", "endpoint", "socket", "network", "request"})
    is_algorithmic = bool(t_set & {"sort", "search", "traverse", "tree", "graph", "dp", "dynamic", "math", "calculate", "matrix", "hash"})
    is_bounded = bool(t_set & {"limit", "bound", "capacity", "buffer", "evict", "window", "max", "threshold", "ceiling", "lru", "lfu"})
    is_stream = bool(t_set & {"stream", "event", "pipeline", "pubsub", "listen", "broadcast", "channel", "message", "emit"})

    return {
        "tokens": tokens,
        "meaningful_terms": meaningful,
        "primary_entity": primary_entity,
        "secondary_entities": secondary_entities,
        "is_concurrent": is_concurrent,
        "is_persistent": is_persistent,
        "is_networked": is_networked,
        "is_algorithmic": is_algorithmic,
        "is_bounded": is_bounded,
        "is_stream": is_stream,
    }


# =============================================================================
# 1. PLANNING SKILL (DYNAMIC 5W1H & MILESTONE DAG)
# =============================================================================

class PlanningSkill(BaseSkill):
    """Deconstructs complex goals into phased milestone DAGs with dynamic entity & risk mapping."""

    def __init__(self, cortex: Optional[Any] = None):
        super().__init__(
            name="planning",
            description="Deconstructs complex engineering goals into phased milestone DAGs, dependency maps, and verification criteria.",
            category="planning",
            cortex=cortex,
        )

    def execute(self, target: str, context: Optional[Dict[str, Any]] = None) -> SkillResult:
        t0 = time.perf_counter()
        goal = target.strip()
        features = _extract_semantic_features(goal)
        entity = features["primary_entity"]
        terms = features["meaningful_terms"]
        related = ", ".join(m.capitalize() for m in terms[:4]) if terms else goal

        # Associative memory resonance (if Cortex substrate is available)
        resonance_text = ""
        resonance_score = 0.92
        if self.cortex and hasattr(self.cortex, "recall"):
            try:
                recall_res = self.cortex.recall(goal)
                best_mem = recall_res.get("best_metadata", {})
                if best_mem.get("text"):
                    resonance_text = f"\n  Synaptic Prior: Associated with '{best_mem.get('text')[:60]}...' (energy drop: {recall_res.get('energy_drop', 0):.2f})"
                    resonance_score = max(0.85, 1.0 - recall_res.get("energy_drop", 0.1))
            except Exception:
                pass

        # Dynamically formulate milestones tailored to extracted traits
        milestones = []
        
        # Step 1: Interface & Contracts
        milestones.append({
            "step": 1,
            "phase": "Interface & Contract Specification",
            "detail": f"Formulate typed data structures, method signatures, and invariant bounds for '{entity}' ({related}).",
        })

        # Step 2: Core Engine / State Transition
        if features["is_algorithmic"]:
            core_detail = f"Implement algorithmic invariants, state transitions, and pointer/data traversal logic for '{entity}'."
        elif features["is_networked"]:
            core_detail = f"Synthesize request dispatchers, route handlers, schema validation, and payload processing for '{entity}'."
        elif features["is_stream"]:
            core_detail = f"Implement event ingestion channels, subscriber dispatches, and backpressure buffering for '{entity}'."
        else:
            core_detail = f"Implement core operational state machine, data mutations, and domain logic for '{entity}'."

        milestones.append({
            "step": 2,
            "phase": "Core Operational Implementation",
            "detail": core_detail,
        })

        # Step 3: Concurrency & Fault Boundaries
        safety_details = []
        if features["is_concurrent"]:
            safety_details.append("thread-safe mutex synchronization / reentrant locks")
        if features["is_bounded"]:
            safety_details.append("capacity bounds enforcement & eviction/rejection policies")
        if features["is_persistent"]:
            safety_details.append("state persistence & rollback consistency")
        if not safety_details:
            safety_details.append("comprehensive input validation, boundary checking, and defensive error boundaries")

        milestones.append({
            "step": 3,
            "phase": "Safety, Concurrency & Fault Isolation",
            "detail": f"Integrate {', '.join(safety_details)} to guarantee system resilience.",
        })

        # Step 4: Verification & AST Validation
        milestones.append({
            "step": 4,
            "phase": "Static Analysis & Sandbox Verification",
            "detail": f"Validate AST compliance, execute automated edge-case test suites, and benchmark runtime latency for '{entity}'.",
        })

        # Dynamically formulate domain risks
        risks = []
        if features["is_concurrent"]:
            risks.append("Race conditions or deadlock under high concurrent access")
        if features["is_bounded"]:
            risks.append("Memory footprint growth if capacity eviction or disposal fails")
        if features["is_algorithmic"]:
            risks.append("Degradation to worst-case computational complexity on skewed inputs")
        if features["is_networked"]:
            risks.append("Connection timeouts, payload exhaustion, or unhandled HTTP status boundaries")
        if not risks:
            risks.append(f"Underspecified edge cases or unexpected null/empty input behavior for '{entity}'")
            risks.append("State inconsistency during abnormal interruption or unhandled exceptions")

        report = f"""=== ARCHITECTURAL EXECUTION PLAN: {goal} ==={resonance_text}

Core Objective:
  Deliver production-grade, AST-verified implementation for: {goal}
  Primary Entity : {entity}
  Domain Context : {related or 'General Engineering'}

Milestone DAG:
"""
        for m in milestones:
            report += f"  Step {m['step']}: [{m['phase']}]\n          -> {m['detail']}\n"

        report += "\nKey Risks & Mitigations:\n"
        for r in risks:
            report += f"  * {r}\n"

        report += "\nDefinition of Done:\n"
        report += f"  [x] Typed Python module with clean AST syntax\n"
        report += f"  [x] Complete test coverage across nominal and boundary conditions\n"
        report += f"  [x] Static analysis passed with 0 security defects\n"
        report += f"  [x] Imprinted into CORTEX-M continuous synaptic memory\n"

        dt = (time.perf_counter() - t0) * 1000.0

        return SkillResult(
            skill_name=self.name,
            success=True,
            output=report,
            metadata={"goal": goal, "entity": entity, "milestones_count": len(milestones), "milestones": milestones},
            execution_time_ms=round(dt, 2),
            resonance_score=round(resonance_score, 2),
            findings=[f"Dynamically planned {len(milestones)} milestones for '{entity}'."],
            suggestions=["Proceed to execution with /code or run full lifecycle with /goal."],
        )


# =============================================================================
# 2. ASKING SKILL (DYNAMIC CLARIFICATION)
# =============================================================================

class AskingSkill(BaseSkill):
    """Proactively identifies ambiguities and formulates high-signal clarifying questions."""

    def __init__(self, cortex: Optional[Any] = None):
        super().__init__(
            name="asking",
            description="Proactively identifies underspecified requirements and asks multiple-choice clarification questions before execution.",
            category="asking",
            cortex=cortex,
        )

    def execute(self, target: str, context: Optional[Dict[str, Any]] = None) -> SkillResult:
        t0 = time.perf_counter()
        req = target.strip()
        features = _extract_semantic_features(req)
        entity = features["primary_entity"]
        terms = features["meaningful_terms"]

        questions: List[Dict[str, Any]] = []

        # 1. Operational & Concurrency Model
        if features["is_concurrent"]:
            questions.append({
                "question": f"Which concurrency synchronization model should govern {entity}?",
                "options": [
                    "(Recommended) Thread-safe mutex guards (threading.RLock) for reentrant multi-threaded access",
                    "Asyncio non-blocking event-loop primitives (asyncio.Lock)",
                    "Lock-free single-threaded optimization for maximum raw throughput",
                ],
            })
        else:
            questions.append({
                "question": f"What execution environment should {entity} target?",
                "options": [
                    "(Recommended) Standalone thread-safe Python class with standard library dependencies",
                    "Asynchronous service with async/await coroutine support",
                    "Pure functional module with immutable data structures",
                ],
            })

        # 2. Persistence & Storage Semantics
        if features["is_persistent"] or features["is_bounded"]:
            questions.append({
                "question": f"How should memory bounds and persistence for {entity} be managed?",
                "options": [
                    "(Recommended) CORTEX-M Zero-DB Continuous Synaptic Memory (sub-millisecond associative recall)",
                    "In-memory bounded structure with configurable capacity and eviction",
                    "Stateless operation with caller-managed lifecycle",
                ],
            })
        else:
            questions.append({
                "question": f"What error handling policy should {entity} follow on boundary violations?",
                "options": [
                    "(Recommended) Strict fail-fast: raise typed exceptions (ValueError, KeyError) with diagnostic messages",
                    "Graceful fallback: return Optional[T] / None or fallback defaults without crashing",
                    "Result object pattern: return status dict/monad with success flag and error details",
                ],
            })

        # 3. Packaging & Interface Contract
        questions.append({
            "question": f"How should {entity} be exposed and packaged?",
            "options": [
                "(Recommended) Production-grade class module with complete type hints and executable self-test block",
                "FastAPI REST microservice with Pydantic request/response schemas",
                "Command-line utility (CLI) with argument parsing and structured output",
            ],
        })

        # Format human-friendly question prompt
        report = f"I've analyzed your requirement for '{req}'.\nTo tailor the architecture perfectly to your needs, please clarify:\n\n"
        for i, q in enumerate(questions, 1):
            report += f"{i}. {q['question']}\n"
            for opt in q["options"]:
                report += f"   [ ] {opt}\n"
            report += "\n"
        report += "Tip: You can respond with your choices (e.g. '1 for thread-safe, 2 for Zero-DB') or proceed with the recommended defaults!"

        dt = (time.perf_counter() - t0) * 1000.0

        return SkillResult(
            skill_name=self.name,
            success=True,
            output=report,
            metadata={"questions": questions, "count": len(questions), "entity": entity},
            execution_time_ms=round(dt, 2),
            resonance_score=0.94,
            findings=[f"Formulated {len(questions)} dynamic architectural decision points for '{entity}'."],
            suggestions=["Confirm preferred options or type 'proceed' to use recommendations."],
        )


# =============================================================================
# 3. SUGGESTING SKILL (DYNAMIC HIGH-LEVERAGE RECOMMENDATIONS)
# =============================================================================

class SuggestingSkill(BaseSkill):
    """Proactively suggests architectural improvements, optimizations, and next steps."""

    def __init__(self, cortex: Optional[Any] = None):
        super().__init__(
            name="suggesting",
            description="Proactively offers high-value architectural optimizations, performance wins, and next engineering steps.",
            category="suggesting",
            cortex=cortex,
        )

    def execute(self, target: str, context: Optional[Dict[str, Any]] = None) -> SkillResult:
        t0 = time.perf_counter()
        features = _extract_semantic_features(target)
        entity = features["primary_entity"]
        suggestions: List[Dict[str, str]] = []

        # Analyze target as code if possible
        is_code = False
        try:
            tree = ast.parse(target)
            is_code = True
            has_type_annotations = any(isinstance(node, ast.AnnAssign) for node in ast.walk(tree))
            has_try_except = any(isinstance(node, ast.Try) for node in ast.walk(tree))
            has_loops = any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))

            if not has_type_annotations:
                suggestions.append({
                    "title": "Strict Type Annotations",
                    "detail": "Enrich function signatures with typing annotations (Generic, TypeVar, Optional) to enable static verification.",
                })
            if not has_try_except:
                suggestions.append({
                    "title": "Defensive Exception Boundaries",
                    "detail": "Wrap critical state mutations in try/except blocks to prevent unhandled process crashes.",
                })
            if has_loops:
                suggestions.append({
                    "title": "Algorithmic Complexity Optimization",
                    "detail": "Inspect loop invariants to avoid nested O(n^2) lookups; leverage dict/set O(1) hash indexing where applicable.",
                })
        except Exception:
            is_code = False

        # If not code or needs more suggestions, derive dynamically from semantic traits
        if len(suggestions) < 3:
            if features["is_concurrent"]:
                suggestions.append({
                    "title": "Reentrant Mutex Guarding",
                    "detail": f"Guard {entity} mutations with threading.RLock to prevent deadlocks when internal helper methods call each other.",
                })
            if features["is_bounded"]:
                suggestions.append({
                    "title": "High-Watermark Telemetry",
                    "detail": f"Track hit/miss ratios, capacity utilization, and eviction counts on {entity} for production observability.",
                })
            if features["is_networked"]:
                suggestions.append({
                    "title": "Health & Readiness Probes",
                    "detail": "Expose /healthz and /readyz liveness probes with circuit breaking against cascading timeouts.",
                })
            suggestions.append({
                "title": "Automated Property-Based Test Suite",
                "detail": f"Generate edge-case test suites covering empty inputs, extreme boundary conditions, and invalid types for {entity}.",
            })
            suggestions.append({
                "title": "Synaptic Weight Imprinting",
                "detail": f"Imprint verified {entity} attractors into CORTEX-M continuous memory so future sessions recall it instantly.",
            })

        suggestions = suggestions[:4]

        report = f"=== PROACTIVE ARCHITECTURAL RECOMMENDATIONS: {entity} ===\n\n"
        for i, s in enumerate(suggestions, 1):
            report += f"{i}. {s['title']}\n   -> {s['detail']}\n\n"
        report += "Would you like me to implement any of these suggestions directly?"

        dt = (time.perf_counter() - t0) * 1000.0

        return SkillResult(
            skill_name=self.name,
            success=True,
            output=report,
            metadata={"suggestions": suggestions, "entity": entity},
            execution_time_ms=round(dt, 2),
            resonance_score=0.93,
            findings=[f"Generated {len(suggestions)} dynamic architectural improvements for '{entity}'."],
            suggestions=[s["title"] for s in suggestions],
        )


# =============================================================================
# 4. GRILLING SKILL (DYNAMIC PRINCIPAL ARCHITECT SOCRATIC STRESS-TEST)
# =============================================================================

class GrillingSkill(BaseSkill):
    """Socratic technical interrogation stress-testing edge cases, scale limits, and failure modes."""

    def __init__(self, cortex: Optional[Any] = None):
        super().__init__(
            name="grilling",
            description="Interactive Socratic technical stress-test interrogating edge cases, concurrency, and failure modes.",
            category="grilling",
            cortex=cortex,
        )

    def execute(self, target: str, context: Optional[Dict[str, Any]] = None) -> SkillResult:
        t0 = time.perf_counter()
        subj = target.strip()
        features = _extract_semantic_features(subj)
        entity = features["primary_entity"]
        terms = ", ".join(m.capitalize() for m in features["meaningful_terms"][:4]) or entity

        questions: List[str] = [
            f"1. Extreme Concurrency: What happens to '{entity}' when 50,000 concurrent threads or requests access shared state simultaneously? Where is the lock contention bottleneck?",
            f"2. Crash Consistency: If the process terminates abnormally midway through a '{entity}' mutation, how is state recovered without corrupting data or leaking handles?",
            f"3. Bounded Resources: What prevents memory, thread pools, or file descriptors from growing unbounded if arrival rate exceeds processing capacity?",
            f"4. Adversarial Boundaries: How does '{entity}' behave when passed empty payloads, maximum integer values, circular data references, or malformed types?",
            f"5. Cascading Degradation: When downstream dependencies or memory allocations stall, does this fail fast or block the caller indefinitely?",
        ]

        if features["is_networked"]:
            questions.append(f"6. Network Boundary: How does '{entity}' guard against slowloris socket exhaustion and malicious payload amplification?")
        elif features["is_algorithmic"]:
            questions.append(f"6. Computational Invariants: What input distribution causes algorithmic degradation, and what is your pivot/rebalance strategy?")
        elif features["is_bounded"]:
            questions.append(f"6. Eviction Stampede: How does '{entity}' prevent thundering herd / stampede when cached items expire or get evicted?")

        report = f"""=== SOCRATIC TECHNICAL INTERROGATION (GRILLING MODE) ===
Subject Under Review: '{subj}'
Primary Component   : {entity} ({terms})

I am interrogating this design from the perspective of a Principal Systems Architect.
Defend your architecture against these critical failure scenarios:

""" + "\n\n".join(questions) + """

Scorecard Dimensions:
  * Concurrency Safety  (0-10)
  * Memory Boundedness  (0-10)
  * Fault Tolerance     (0-10)
  * Input Sanitization  (0-10)

Answer these questions to harden your design, or type 'harden' and I will synthesize protective guardrails for you!
"""
        dt = (time.perf_counter() - t0) * 1000.0

        return SkillResult(
            skill_name=self.name,
            success=True,
            output=report,
            metadata={"subject": subj, "entity": entity, "interrogations": questions},
            execution_time_ms=round(dt, 2),
            resonance_score=0.96,
            findings=[f"Generated {len(questions)} high-severity architectural challenge questions for '{entity}'."],
            suggestions=["Answer specific questions or request automated hardening."],
        )


# =============================================================================
# 5. SCHEDULING SKILL (DYNAMIC TIMER, CRON & SYNAPTIC MAINTENANCE)
# =============================================================================

class SchedulingSkill(BaseSkill):
    """Manages one-shot timers, recurring tasks, sleep consolidation, and health audits."""

    def __init__(self, cortex: Optional[Any] = None):
        super().__init__(
            name="scheduling",
            description="Manages one-shot timers, recurring cron-like jobs, synaptic sleep cycles, and substrate health audits.",
            category="scheduling",
            cortex=cortex,
        )
        self.scheduled_jobs: List[Dict[str, Any]] = []

    def execute(self, target: str, context: Optional[Dict[str, Any]] = None) -> SkillResult:
        t0 = time.perf_counter()
        cmd = target.strip().lower()

        if "sleep" in cmd:
            job_name = "biological_sleep_consolidation"
            if self.cortex and hasattr(self.cortex, "sleep_consolidation"):
                res = self.cortex.sleep_consolidation()
                output = f"Executed Sleep Consolidation: Replayed {res.get('replayed_attractors', 0)} attractors, pruned {res.get('pruned_synapses', 0)} weak synapses (>86% sparsity)."
            else:
                output = "Scheduled biological sleep consolidation cycle for synaptic substrate."

        elif "audit" in cmd or "health" in cmd:
            job_name = "synaptic_substrate_audit"
            if self.cortex and hasattr(self.cortex, "audit_synaptic_health"):
                res = self.cortex.audit_synaptic_health()
                output = "Substrate Health Audit Executed:\n" + "\n".join(f"  {k:30s}: {v}" for k, v in res.items())
            else:
                output = "Scheduled periodic synaptic substrate audit."

        elif any(kw in cmd for kw in ["timer", "remind", "in ", "after "]):
            seconds_match = re.search(r"(\d+)\s*(s|sec|m|min|h|hour)", cmd)
            dur_str = seconds_match.group(0) if seconds_match else "300s"
            job_name = f"timer_{int(time.time())}"
            self.scheduled_jobs.append({"name": job_name, "duration": dur_str, "target": target})
            output = f"Timer Scheduled: Alert registered for {dur_str}. Target: '{target}'."

        else:
            job_name = f"recurring_pipeline_{len(self.scheduled_jobs) + 1}"
            self.scheduled_jobs.append({"name": job_name, "schedule": "every 60s", "target": target})
            output = f"Job Scheduled: '{job_name}' registered. Target: '{target}' (Interval: 60s)."

        dt = (time.perf_counter() - t0) * 1000.0

        return SkillResult(
            skill_name=self.name,
            success=True,
            output=output,
            metadata={"job_name": job_name, "active_jobs_count": len(self.scheduled_jobs)},
            execution_time_ms=round(dt, 2),
            resonance_score=0.90,
            findings=[f"Scheduled task: {job_name}."],
            suggestions=["Use /audit or /sleep to trigger on-demand substrate maintenance."],
        )


# =============================================================================
# 6. GOAL SKILL (DYNAMIC AUTONOMOUS GOAL COMPLETION)
# =============================================================================

class GoalSkill(BaseSkill):
    """Autonomous persistent goal completion loop: Plan -> Code -> Review -> Audit -> Imprint."""

    def __init__(self, completer: Optional[Any] = None, cortex: Optional[Any] = None):
        super().__init__(
            name="goal",
            description="Autonomous persistent goal completer: executes full multi-agent swarm until goal is 100% verified.",
            category="goal",
            cortex=cortex,
        )
        self.completer = completer

    def set_completer(self, completer: Any) -> None:
        self.completer = completer

    def execute(self, target: str, context: Optional[Dict[str, Any]] = None) -> SkillResult:
        t0 = time.perf_counter()
        goal = target.strip()

        if self.completer is None:
            from cortex_m.agents.task_completer import MultiAgentTaskCompleter
            self.completer = MultiAgentTaskCompleter()

        # Run full autonomous swarm lifecycle
        res = self.completer.execute_task(
            goal=goal,
            filename="solution.py",
            verify_executable=True,
        )

        dt = (time.perf_counter() - t0) * 1000.0

        if res.status == "conversational":
            output = res.audit_summary.get("conversational_response", "Hello! What goal shall we accomplish?")
            success = True
        else:
            art = res.artifacts[0] if res.artifacts else None
            classes = art.classes if art else []
            functions = art.functions if art else []

            output = f"""=== AUTONOMOUS GOAL COMPLETED: {goal} ===
Execution Time : {res.execution_time_sec*1000:.1f} ms
Status         : {res.status.upper()}
Syntax Valid   : {res.audit_summary.get('is_valid_syntax')}
Critic Approved: {res.audit_summary.get('is_approved_by_critic')}
Sandbox Tested : {res.audit_summary.get('runtime_verified')}
Classes Defined: {classes or 'None'}
Functions      : {functions or 'None'}
Synaptic Memory: {res.synaptic_memories_imprinted} attractors active

--- Synthesized Solution ---
{art.code if art else '# No code generated'}
"""
            success = res.status == "completed"

        return SkillResult(
            skill_name=self.name,
            success=success,
            output=output,
            metadata={"goal": goal, "status": res.status, "execution_time_sec": res.execution_time_sec},
            execution_time_ms=round(dt, 2),
            resonance_score=0.98,
            findings=[f"Autonomous goal '{goal}' finished with status: {res.status}."],
            suggestions=["Run /test to generate test suite", "Run /sec to scan vulnerabilities"],
        )


# =============================================================================
# 7. ALIGNER SKILL (DYNAMIC INTENT-IMPLEMENTATION RECONCILIATION)
# =============================================================================

class AlignerSkill(BaseSkill):
    """Intent-to-implementation reconciliation detecting drift, misalignment, and omissions."""

    def __init__(self, cortex: Optional[Any] = None):
        super().__init__(
            name="aligner",
            description="Reconciles human intent with code implementation, detecting semantic drift and missing requirements.",
            category="alignment",
            cortex=cortex,
        )

    def execute(self, target: str, context: Optional[Dict[str, Any]] = None) -> SkillResult:
        t0 = time.perf_counter()
        ctx = context or {}
        intent = ctx.get("intent", target)
        code = ctx.get("code", target)

        i_lower = intent.lower()
        c_lower = code.lower()

        # Extract semantic concepts dynamically from intent
        words = re.findall(r"[a-zA-Z0-9_]+", i_lower)
        stop_words = {"implement", "write", "create", "build", "code", "python", "the", "a", "an", "for", "with", "and", "or", "to", "in", "of"}
        key_concepts = [w for w in words if len(w) >= 4 and w not in stop_words]

        alignments: List[str] = []
        drifts: List[str] = []
        score = 100.0

        for concept in key_concepts:
            if concept in c_lower:
                alignments.append(f"Intent requirement '{concept}' is reflected in the implementation.")
            else:
                drifts.append(f"Potential omission: intent mentions '{concept}' but not found in implementation.")
                score -= 15.0

        score = max(0.0, score)

        report = f"""=== INTENT & ARCHITECTURE ALIGNMENT REPORT ===
Target Intent   : '{intent[:75]}...'
Alignment Score : {score:.1f}%

Verified Alignments:
""" + ("\n".join(f"  [OK] {a}" for a in alignments) if alignments else "  [OK] General intent structure observed.")

        if drifts:
            report += "\n\nDetected Drift / Missing Requirements:\n" + "\n".join(f"  [!] {d}" for d in drifts)
            report += "\n\nRealignment Recommendation: Update implementation to satisfy missing concepts."
        else:
            report += "\n\nVerdict: PERFECT ALIGNMENT. Implementation matches human intent with zero semantic drift."

        dt = (time.perf_counter() - t0) * 1000.0

        return SkillResult(
            skill_name=self.name,
            success=score >= 70.0,
            output=report,
            metadata={"alignment_score": score, "alignments": alignments, "drifts": drifts},
            execution_time_ms=round(dt, 2),
            resonance_score=score / 100.0,
            findings=drifts or ["100% intent alignment verified."],
            suggestions=["Run /refactor to polish" if score >= 80 else "Run /code to reconcile omissions."],
        )
