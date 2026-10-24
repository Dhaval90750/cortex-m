"""CortexCode (CORTEX-Coder): State-of-the-Art Autonomous Industrial Coding CLI.

Powered by CORTEX-M (Context-Oriented Routing and Trace Exchange for MIMO).
Features:
- Multi-agent swarm (Architect, Coder, Critic, Auditor, Coordinator)
- Zero-database continuous Synaptic Blackboard for agent consensus
- Dynamic code synthesis (zero hardcoded stubs) with AST static analysis
- Sandbox runtime execution and autonomous self-repair
- Human-centric conversational intelligence & architectural advice
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

# ANSI color styling for modern terminals
CYAN = "\033[1;36m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
MAGENTA = "\033[1;35m"
BLUE = "\033[1;34m"
RED = "\033[1;31m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

BANNER = fr"""
{CYAN}================================================================================{RESET}
{BOLD}   ______           __              ______          __{RESET}
{BOLD}  / ____/___  _____/ /____  _  __  / ____/___  ____/ /__ {RESET}
{BOLD} / /   / __ \/ ___/ __/ _ \| |/_/ / /   / __ \/ __  / _ \{RESET}
{BOLD}/ /___/ /_/ / /  / /_/  __/>  <  / /___/ /_/ / /_/ /  __/{RESET}
{BOLD}\____/\____/_/   \__/\___/_/|_|  \____/\____/\__,_/\___/ {RESET}
{GREEN}  CortexCode (CORTEX-Coder) | Industrial Multi-Agent Coding Swarm{RESET}
{DIM}  Context-Oriented Routing and Trace Exchange for MIMO (CORTEX-M){RESET}
{CYAN}================================================================================{RESET}
  {MAGENTA}[Architect]{RESET} : 5W1H decomposition, milestone DAG, and interface design
  {BLUE}[Coder]{RESET}     : Dynamic, production-grade, type-annotated code with AST validation
  {YELLOW}[Critic]{RESET}    : Static analysis, security risk auditing (eval/exec detection)
  {GREEN}[Auditor]{RESET}   : Isolated runtime execution and autonomous self-repair loops
  {CYAN}[Synaptic]{RESET}  : Zero-database agent consensus via continuous Hopfield attractors
{CYAN}================================================================================{RESET}
{DIM}Commands:{RESET}
  <task description>      -> Full multi-agent swarm lifecycle (Plan -> Code -> Review -> Audit)
  /plan <goal>            -> Architect agent plans architecture and milestones
  /code <task>            -> Coder agent synthesizes code with AST verification
  /review <file>          -> Critic agent performs static analysis & security check
  /blackboard             -> Inspect active agent memories on the Synaptic Blackboard
  /audit                  -> Display substrate capacity and synaptic health
  /help                   -> Display command reference
  /exit                   -> Quit
{CYAN}================================================================================{RESET}
"""


def run_interactive(completer: MultiAgentTaskCompleter):
    print(BANNER)
    print(f"{GREEN}[CortexCode]{RESET} Autonomous industrial coding agent initialized and ready.")
    print("Talk to me naturally, ask architectural questions, or give me any coding goal!\n")

    while True:
        try:
            user_input = input(f"{BOLD}{CYAN}cortex-code >> {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{YELLOW}Exiting CortexCode. Synaptic state preserved.{RESET}")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("/exit", "/quit", "exit", "quit"):
            print(f"{YELLOW}Exiting CortexCode. Goodbye!{RESET}")
            break
        elif cmd in ("/help", "help"):
            print(BANNER)
            continue
        elif cmd.startswith("/plan "):
            goal = user_input[6:].strip()
            print(f"\n{MAGENTA}[Architect Agent]{RESET} Deconstructing goal: '{BOLD}{goal}{RESET}'...")
            t0 = time.perf_counter()
            plan = completer.architect.plan(goal)
            dt = time.perf_counter() - t0
            print(f"\n{BOLD}Architecture Blueprint:{RESET}\n{plan['architecture_overview']}\n")
            print(f"{BOLD}Execution Milestones ({len(plan['milestones'])} steps):{RESET}")
            for m in plan["milestones"]:
                print(f"  {CYAN}Step {m['step']}:{RESET} {BOLD}{m['name']}{RESET} — {m['desc']}")
            print(f"\n{DIM}[Deconstructed in {dt*1000:.1f}ms | Synaptic Resonance: {plan['resonance']*100:.0f}%]{RESET}\n")
        elif cmd.startswith("/code "):
            task = user_input[6:].strip()
            print(f"\n{BLUE}[Coder Agent]{RESET} Synthesizing code for: '{BOLD}{task}{RESET}'...")
            t0 = time.perf_counter()
            art = completer.coder.synthesize_code(task, filename="solution.py")
            dt = time.perf_counter() - t0
            print(f"  Syntax Valid: {GREEN if art.is_valid_syntax else RED}{art.is_valid_syntax}{RESET}")
            print(f"  Classes     : {art.classes or 'None'}")
            print(f"  Functions   : {art.functions or 'None'}")
            print(f"  Imports     : {art.imports or 'None'}")
            print(f"\n{CYAN}{'=' * 68}{RESET}")
            print(art.code)
            print(f"{CYAN}{'=' * 68}{RESET}")
            print(f"{DIM}[Synthesized in {dt*1000:.1f}ms | AST Validated]{RESET}\n")
        elif cmd.startswith("/review "):
            fname = user_input[8:].strip()
            if os.path.exists(fname):
                with open(fname, "r", encoding="utf-8") as f:
                    code_str = f.read()
                art = completer.coder._analyze_code_ast(fname, code_str, "python")
                review = completer.critic.review_code(art)
                print(f"\n{YELLOW}[Critic Agent]{RESET} Reviewing '{BOLD}{fname}{RESET}':")
                print(f"  Approved    : {GREEN if review.is_approved else RED}{review.is_approved}{RESET}")
                print(f"  Risk Score  : {review.risk_score:.2f} (0=safe, 1=critical defects)")
                if review.issues_found:
                    print(f"  {RED}Issues Found:{RESET}")
                    for issue in review.issues_found:
                        print(f"    - {issue}")
                if review.suggested_improvements:
                    print(f"  {YELLOW}Suggestions:{RESET}")
                    for sug in review.suggested_improvements:
                        print(f"    - {sug}")
                print()
            else:
                print(f"{RED}[Error]{RESET} File not found: {fname}\n")
        elif cmd in ("/blackboard", "/board"):
            msgs = list(completer.blackboard.messages.values())
            print(f"\n{CYAN}[Synaptic Blackboard]{RESET} {BOLD}{len(msgs)}{RESET} active agent memories:")
            for m in msgs[-10:]:
                role_color = {
                    AgentRole.ARCHITECT: MAGENTA,
                    AgentRole.CODER: BLUE,
                    AgentRole.CRITIC: YELLOW,
                    AgentRole.AUDITOR: GREEN,
                    AgentRole.COORDINATOR: CYAN,
                }.get(m.sender_role, RESET)
                print(f"  {role_color}[{m.sender_role.value.upper()}]{RESET} ({m.topic}) {m.content[:75]}...")
            print()
        elif cmd == "/audit":
            audit = completer.lm.cortex.audit_synaptic_health()
            print(f"\n{CYAN}--- Physical Synaptic Substrate Audit ---{RESET}")
            for k, v in audit.items():
                print(f"  {k:30s}: {BOLD}{v}{RESET}")
            print(f"{CYAN}-----------------------------------------{RESET}\n")
        else:
            # Full autonomous multi-agent task execution
            t0 = time.perf_counter()
            res = completer.execute_task(
                goal=user_input,
                filename="solution.py",
                verify_executable=True,
            )
            dt = time.perf_counter() - t0

            # 1. Conversational / Human dialogue response
            if res.status == "conversational":
                reply = res.audit_summary.get("conversational_response", "How can I help you code today?")
                print(f"\n{GREEN}CortexCode>{RESET} {reply}\n")
                continue

            # 2. Completed Multi-Agent Coding Workflow
            print(f"\n{CYAN}----------------------------------------------------------------------{RESET}")
            print(f"{BOLD}[Multi-Agent Swarm Report]{RESET} Task '{BOLD}{user_input}{RESET}' completed in {dt*1000:.1f}ms")
            print(f"  {MAGENTA}[Architect]{RESET}  Milestone DAG planned ({len(res.messages)} blackboard traces)")
            print(f"  {BLUE}[Coder]{RESET}      Synthesized solution ({len(res.artifacts[0].code.splitlines()) if res.artifacts else 0} lines)")
            print(f"  {YELLOW}[Critic]{RESET}     AST Static Analysis & Security Approved: {GREEN}{res.audit_summary.get('is_approved_by_critic')}{RESET}")
            print(f"  {GREEN}[Auditor]{RESET}    Runtime Execution Sandbox Tested: {GREEN}{res.audit_summary.get('runtime_verified')}{RESET}")
            print(f"  {CYAN}[Memory]{RESET}     Continuous Synapses Imprinted: {BOLD}{res.synaptic_memories_imprinted}{RESET} attractors")
            print(f"{CYAN}----------------------------------------------------------------------{RESET}")

            if res.artifacts:
                art = res.artifacts[0]
                print(f"\n{BOLD}=== GENERATED CODE: {art.filename} ==={RESET}")
                print(art.code)
                print(f"{BOLD}========================================{RESET}")
                print(f"{DIM}Classes: {art.classes or 'None'} | Functions: {art.functions or 'None'}{RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="CortexCode (CORTEX-Coder) Industrial CLI")
    parser.add_argument("--goal", type=str, help="Coding goal to complete in one-shot mode")
    parser.add_argument("--output", type=str, default="solution.py", help="Output filename for code")
    args = parser.parse_args()

    completer = MultiAgentTaskCompleter()

    if args.goal:
        res = completer.execute_task(goal=args.goal, filename=args.output, verify_executable=True)
        if res.status == "conversational":
            print(res.audit_summary.get("conversational_response", ""))
        else:
            print(f"[CortexCode] Completed '{args.goal}' in {res.execution_time_sec*1000:.1f}ms")
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
