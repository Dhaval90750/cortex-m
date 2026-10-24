"""Interactive Terminal CLI for CortexCode (CORTEX-Coder) & Multi-Agent Swarm.

Run natively with:
    python scripts/cortex_code.py

Or one-shot execution:
    python scripts/cortex_code.py --goal "Implement an LRU cache" --output lru_cache.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure src is in python path
sys.path.insert(0, "src")

from cortex_m.agents.blackboard import AgentRole, SynapticBlackboard
from cortex_m.agents.coding_agent import CortexCodingAgent
from cortex_m.agents.task_completer import MultiAgentTaskCompleter
from cortex_m.model.cortex_lm import CortexLM

BANNER = r"""
================================================================================
   ______           __              ______          __
  / ____/___  _____/ /____  _  __  / ____/___  ____/ /__ 
 / /   / __ \/ ___/ __/ _ \| |/_/ / /   / __ \/ __  / _ \
/ /___/ /_/ / /  / /_/  __/>  <  / /___/ /_/ / /_/ /  __/
\____/\____/_/   \__/\___/_/|_|  \____/\____/\__,_/\___/ 
  CortexCode (CORTEX-Coder) | Autonomous Industrial Coding Agent
  Context-Oriented Routing and Trace Exchange for MIMO (CORTEX-M)
================================================================================
  * Architect Agent : 5W1H decomposition, milestone DAG, and interface design
  * Coder Agent     : Idiomatic, type-annotated code with AST syntax validation
  * Critic Agent    : Static analysis, security risk auditing (eval/exec detection)
  * Auditor Agent   : Isolated runtime execution and autonomous self-repair loops
  * Synaptic Board  : Zero-database agent consensus via continuous Hopfield attractors
================================================================================
Commands:
  <task description>      -> Full multi-agent swarm lifecycle (Plan -> Code -> Review -> Audit)
  /plan <goal>            -> Architect agent plans architecture and milestones
  /code <task>            -> Coder agent synthesizes code with AST verification
  /review <file>          -> Critic agent performs static analysis & security check
  /blackboard             -> Inspect all active messages on the Synaptic Blackboard
  /audit                  -> Display substrate capacity and synaptic health
  /help                   -> Display command reference
  /exit                   -> Quit
================================================================================
"""


def run_interactive(completer: MultiAgentTaskCompleter):
    print(BANNER)
    print("[CortexCode] Autonomous industrial coding agent initialized and ready.")
    print("Type any coding task or command below (or '/exit' to quit):\n")

    while True:
        try:
            user_input = input("cortex-code >> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting CortexCode.")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("/exit", "/quit", "exit", "quit"):
            print("Exiting CortexCode. Goodbye!")
            break
        elif cmd in ("/help", "help"):
            print(BANNER)
            continue
        elif cmd.startswith("/plan "):
            goal = user_input[6:].strip()
            print(f"\n[Architect Agent] Deconstructing goal: '{goal}'...")
            t0 = time.perf_counter()
            plan = completer.architect.plan(goal)
            dt = time.perf_counter() - t0
            print(f"  Architecture: {plan['architecture_overview']}")
            print(f"  Milestones ({len(plan['milestones'])} steps):")
            for m in plan["milestones"]:
                print(f"    Step {m['step']}: {m['name']} - {m['desc']}")
            print(f"  [Time: {dt*1000:.1f}ms | Resonance: {plan['resonance']*100:.0f}%]\n")
        elif cmd.startswith("/code "):
            task = user_input[6:].strip()
            print(f"\n[Coder Agent] Synthesizing code for: '{task}'...")
            t0 = time.perf_counter()
            art = completer.coder.synthesize_code(task, filename="generated.py")
            dt = time.perf_counter() - t0
            print(f"  Syntax Valid: {art.is_valid_syntax}")
            print(f"  Classes     : {art.classes or 'None'}")
            print(f"  Functions   : {art.functions or 'None'}")
            print(f"  Imports     : {art.imports or 'None'}")
            print("-" * 60)
            print(art.code)
            print("-" * 60)
            print(f"  [Time: {dt*1000:.1f}ms]\n")
        elif cmd.startswith("/review "):
            fname = user_input[8:].strip()
            if os.path.exists(fname):
                with open(fname, "r", encoding="utf-8") as f:
                    code_str = f.read()
                art = completer.coder._analyze_code_ast(fname, code_str, "python")
                review = completer.critic.review_code(art)
                print(f"\n[Critic Agent] Reviewing '{fname}':")
                print(f"  Approved    : {review.is_approved}")
                print(f"  Risk Score  : {review.risk_score:.2f} (0=safe, 1=critical)")
                if review.issues_found:
                    print(f"  Issues      : {review.issues_found}")
                if review.suggested_improvements:
                    print(f"  Suggestions : {review.suggested_improvements}")
                print()
            else:
                print(f"[Error] File not found: {fname}\n")
        elif cmd in ("/blackboard", "/board"):
            msgs = list(completer.blackboard.messages.values())
            print(f"\n[Synaptic Blackboard] {len(msgs)} active agent memories:")
            for m in msgs[-10:]:
                print(f"  [{m.sender_role.value.upper()}] ({m.topic}) {m.content[:75]}...")
            print()
        elif cmd == "/audit":
            audit = completer.lm.cortex.audit_synaptic_health()
            print("\n[Synaptic Substrate Audit]")
            for k, v in audit.items():
                print(f"  {k:28}: {v}")
            print()
        else:
            # Full autonomous multi-agent task completion
            print(f"\n[CortexCode] Executing task: '{user_input}'...")
            t0 = time.perf_counter()
            res = completer.execute_task(
                goal=user_input,
                filename="solution.py",
                verify_executable=True,
            )
            dt = time.perf_counter() - t0

            print(f"  Status          : {res.status.upper()}")
            print(f"  Execution Time  : {res.execution_time_sec*1000:.1f} ms")
            print(f"  Memories Learned: {res.synaptic_memories_imprinted} attractor basins")
            print(f"  Syntax Valid    : {res.audit_summary.get('is_valid_syntax')}")
            print(f"  Runtime Tested  : {res.audit_summary.get('runtime_verified')}")

            if res.artifacts:
                art = res.artifacts[0]
                print("\n" + "=" * 60)
                print(f"--- GENERATED CODE: {art.filename} ---")
                print("=" * 60)
                print(art.code)
                print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="CortexCode (CORTEX-Coder) Industrial CLI")
    parser.add_argument("--goal", type=str, help="Coding goal to complete in one-shot mode")
    parser.add_argument("--output", type=str, default="solution.py", help="Output filename for code")
    args = parser.parse_args()

    completer = MultiAgentTaskCompleter()

    if args.goal:
        print(f"[CortexCode] One-Shot Execution: '{args.goal}'")
        res = completer.execute_task(goal=args.goal, filename=args.output, verify_executable=True)
        print(f"Status: {res.status} in {res.execution_time_sec*1000:.1f}ms")
        if res.artifacts:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(res.artifacts[0].code)
            print(f"Saved generated code to: {args.output}")
            print("-" * 60)
            print(res.artifacts[0].code)
            print("-" * 60)
    else:
        run_interactive(completer)


if __name__ == "__main__":
    main()
