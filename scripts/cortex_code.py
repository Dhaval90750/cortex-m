"""CortexCode (CORTEX-Coder): State-of-the-Art Autonomous Industrial Coding CLI.

Powered by CORTEX-M (Context-Oriented Routing and Trace Exchange for MIMO).
Features:
- Multi-agent swarm (Architect, Coder, Critic, Auditor, Coordinator)
- Zero-database continuous Synaptic Blackboard for agent consensus
- Modular Cognitive Skills (/plan, /ask, /suggest, /grill, /schedule, /goal, /align)
- Software Engineering Skills (/sec, /test, /refactor, /bigo, /api, /docs, /sql)
- Dynamic code synthesis with zero dummy stubs and full AST compliance
- Sandbox runtime execution and autonomous self-repair loops
- Human-centric conversational intelligence & architectural advice
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure src is in python path regardless of where the command is invoked from
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from cortex_m.agents.blackboard import AgentRole, SynapticBlackboard
from cortex_m.agents.coding_agent import CortexCodingAgent
from cortex_m.agents.task_completer import MultiAgentTaskCompleter
from cortex_m.agents.terminal_ux import render_markdown, swarm_progress
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
{DIM}Agentic Cognitive Skills:{RESET}
  /plan <goal>            -> Deconstruct goal into phased milestone DAG & risk map
  /ask <requirement>      -> Proactively clarify underspecified requirements
  /suggest <topic/code>   -> Proactive high-leverage architectural optimizations
  /grill <subject/code>   -> Socratic technical stress-test by Principal Architect
  /schedule <timer/sleep> -> Timers, recurring cron jobs, or sleep consolidation
  /goal <task>            -> Autonomous end-to-end swarm execution & verification
  /align <target>         -> Reconcile implementation against human intent
{DIM}Software Engineering Skills:{RESET}
  /sec <file/code>        -> Fast security vulnerability & injection scan
  /test <file/code>       -> Generate complete runnable unit test suite
  /refactor <file/code>   -> Modernize, clean, and type-annotate code
  /bigo <file/code>       -> Theoretical Big-O time & space complexity analysis
  /api <resource>         -> Scaffold production FastAPI CRUD microservice
  /docs <file/code>       -> Generate structured docstrings and specs
  /sql <query>            -> Analyze SQL queries for index optimization
{DIM}System Commands:{RESET}
  <task description>      -> Autonomous swarm loop (Plan -> Code -> Review -> Audit)
  /skills                 -> List all registered cognitive & engineering skills
  /blackboard             -> Inspect active agent memories on the Synaptic Blackboard
  /audit                  -> Display substrate capacity and synaptic health
  /help                   -> Display command reference
  /exit                   -> Quit
{CYAN}================================================================================{RESET}
"""


def _resolve_target(arg: str) -> str:
    """If arg points to an existing file, read its content; otherwise return arg."""
    arg_clean = arg.strip()
    if os.path.exists(arg_clean) and os.path.isfile(arg_clean):
        try:
            with open(arg_clean, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return arg_clean
    return arg_clean


def run_interactive(completer: MultiAgentTaskCompleter):
    print(BANNER)
    print(f"{GREEN}[CortexCode]{RESET} Autonomous industrial coding agent initialized and ready.")
    print("Talk to me naturally, invoke skills (/plan, /ask, /suggest, /grill), or give me any coding goal!\n")

    while True:
        try:
            user_input = input(f"{BOLD}{CYAN}cortex-code >> {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{YELLOW}Exiting CortexCode. Synaptic state preserved.{RESET}")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        # Exit
        if cmd in ("/exit", "/quit", "exit", "quit"):
            print(f"{YELLOW}Exiting CortexCode. Goodbye!{RESET}")
            break

        # Help
        elif cmd in ("/help", "help"):
            print(BANNER)
            continue

        # Skills listing
        elif cmd in ("/skills", "skills"):
            cog = completer.skills.list_cognitive_skills()
            eng = completer.skills.list_engineering_skills()
            print(f"\n{BOLD}{CYAN}=== CORTEX-M REGISTERED SKILLS SUITE ==={RESET}\n")
            print(f"{BOLD}{MAGENTA}[Agentic Cognitive Skills]{RESET}")
            for s in cog:
                aliases_str = f" [aliases: {', '.join(s['aliases'])}]" if s["aliases"] else ""
                print(f"  {CYAN}/{s['name']:12s}{RESET}: {s['description']}{DIM}{aliases_str}{RESET}")
            print(f"\n{BOLD}{BLUE}[Software Engineering & Analysis Skills]{RESET}")
            for s in eng:
                aliases_str = f" [aliases: {', '.join(s['aliases'])}]" if s["aliases"] else ""
                print(f"  {CYAN}/{s['name']:12s}{RESET}: {s['description']}{DIM}{aliases_str}{RESET}")
            print(f"\n{DIM}Run any skill using: /{BOLD}<skill_name>{RESET}{DIM} <target> or naturally: '<skill_name> <target>'{RESET}\n")
            continue

        # Cognitive: /plan
        elif cmd.startswith("/plan ") or cmd.startswith("plan "):
            prefix_len = 6 if cmd.startswith("/plan ") else 5
            goal = user_input[prefix_len:].strip()
            print(f"\n{MAGENTA}[Planning Skill]{RESET} Formulating dynamic architectural blueprint for: '{BOLD}{goal}{RESET}'...")
            res = completer.execute_skill("planning", goal)
            print(f"\n{res.output}\n")
            print(f"{DIM}[Dynamic planning completed in {res.execution_time_ms:.1f}ms | Resonance: {res.resonance_score*100:.0f}%]{RESET}\n")
            continue

        # Cognitive: /ask
        elif cmd.startswith("/ask ") or cmd.startswith("ask ") or cmd.startswith("/clarify ") or cmd.startswith("clarify "):
            if cmd.startswith("/ask "):
                req = user_input[5:].strip()
            elif cmd.startswith("ask "):
                req = user_input[4:].strip()
            elif cmd.startswith("/clarify "):
                req = user_input[9:].strip()
            else:
                req = user_input[8:].strip()
            print(f"\n{CYAN}[Asking Skill]{RESET} Analyzing ambiguities & decision points for: '{BOLD}{req}{RESET}'...")
            res = completer.execute_skill("asking", req)
            print(f"\n{res.output}\n")
            continue

        # Cognitive: /suggest
        elif cmd.startswith("/suggest ") or cmd.startswith("suggest ") or cmd.startswith("/recommend "):
            target = user_input.split(" ", 1)[1].strip()
            content = _resolve_target(target)
            print(f"\n{YELLOW}[Suggesting Skill]{RESET} Deriving high-leverage architectural optimizations...")
            res = completer.execute_skill("suggesting", content)
            print(f"\n{res.output}\n")
            continue

        # Cognitive: /grill
        elif cmd.startswith("/grill ") or cmd.startswith("grill "):
            prefix_len = 7 if cmd.startswith("/grill ") else 6
            subj = user_input[prefix_len:].strip()
            content = _resolve_target(subj)
            print(f"\n{RED}[Grilling Skill]{RESET} Initiating Principal Architect Socratic Interrogation...")
            res = completer.execute_skill("grilling", content)
            print(f"\n{res.output}\n")
            continue

        # Cognitive: /schedule
        elif cmd.startswith("/schedule ") or cmd.startswith("schedule "):
            prefix_len = 10 if cmd.startswith("/schedule ") else 9
            target = user_input[prefix_len:].strip()
            print(f"\n{CYAN}[Scheduling Skill]{RESET} Scheduling execution...")
            res = completer.execute_skill("scheduling", target)
            print(f"\n{res.output}\n")
            continue

        # Cognitive: /goal
        elif cmd.startswith("/goal ") or cmd.startswith("goal "):
            prefix_len = 6 if cmd.startswith("/goal ") else 5
            task = user_input[prefix_len:].strip()
            print(f"\n{GREEN}[Goal Skill]{RESET} Executing autonomous multi-agent swarm for: '{BOLD}{task}{RESET}'...")
            res = completer.execute_skill("goal", task)
            print(f"\n{res.output}\n")
            continue

        # Cognitive: /align
        elif cmd.startswith("/align ") or cmd.startswith("align "):
            prefix_len = 7 if cmd.startswith("/align ") else 6
            target = user_input[prefix_len:].strip()
            print(f"\n{BLUE}[Aligner Skill]{RESET} Reconciling intent vs implementation...")
            res = completer.execute_skill("aligner", target)
            print(f"\n{res.output}\n")
            continue

        # Engineering: /sec
        elif cmd.startswith("/sec ") or cmd.startswith("/security "):
            target = user_input.split(" ", 1)[1].strip()
            content = _resolve_target(target)
            print(f"\n{YELLOW}[Security Audit Skill]{RESET} Scanning for injection, AST vulnerabilities, and hardcoded secrets...")
            res = completer.execute_skill("security", content)
            print(f"\n{res.output}\n")
            continue

        # Engineering: /test
        elif cmd.startswith("/test ") or cmd.startswith("/tests "):
            target = user_input.split(" ", 1)[1].strip()
            content = _resolve_target(target)
            print(f"\n{GREEN}[Test Generation Skill]{RESET} Synthesizing runnable unit test suite...")
            res = completer.execute_skill("test", content)
            print(f"\n{res.output}\n")
            continue

        # Engineering: /refactor
        elif cmd.startswith("/refactor "):
            target = user_input[10:].strip()
            content = _resolve_target(target)
            print(f"\n{BLUE}[Refactoring Skill]{RESET} Modernizing and type-annotating code...")
            res = completer.execute_skill("refactor", content)
            print(f"\n{res.output}\n")
            continue

        # Engineering: /bigo
        elif cmd.startswith("/bigo ") or cmd.startswith("/complexity "):
            target = user_input.split(" ", 1)[1].strip()
            content = _resolve_target(target)
            print(f"\n{MAGENTA}[Complexity Analysis Skill]{RESET} Analyzing theoretical Big-O time and space...")
            res = completer.execute_skill("complexity", content)
            print(f"\n{res.output}\n")
            continue

        # Engineering: /api
        elif cmd.startswith("/api "):
            resource = user_input[5:].strip()
            print(f"\n{CYAN}[API Scaffolding Skill]{RESET} Generating production FastAPI CRUD router for '{BOLD}{resource}{RESET}'...")
            res = completer.execute_skill("api", resource)
            print(f"\n{res.output}\n")
            continue

        # Engineering: /docs
        elif cmd.startswith("/docs "):
            target = user_input[6:].strip()
            content = _resolve_target(target)
            print(f"\n{CYAN}[Documentation Skill]{RESET} Synthesizing Google-style docstrings and specs...")
            res = completer.execute_skill("docs", content)
            print(f"\n{res.output}\n")
            continue

        # Engineering: /sql
        elif cmd.startswith("/sql "):
            query = user_input[5:].strip()
            print(f"\n{YELLOW}[SQL Optimization Skill]{RESET} Profiling query and indexing strategy...")
            res = completer.execute_skill("sql", query)
            print(f"\n{res.output}\n")
            continue

        # Generic modular skill invocation: /skill <name> <target>
        elif cmd.startswith("/skill "):
            parts = user_input[7:].strip().split(" ", 1)
            s_name = parts[0]
            s_target = parts[1] if len(parts) > 1 else ""
            content = _resolve_target(s_target)
            res = completer.execute_skill(s_name, content)
            print(f"\n{res.output}\n")
            continue

        # Code direct command: /code <task>
        elif cmd.startswith("/code "):
            task = user_input[6:].strip()
            print(f"\n{BLUE}[Coder Agent]{RESET} Synthesizing dynamic code for: '{BOLD}{task}{RESET}'...")
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
            continue

        # Review direct command: /review <file>
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
            continue

        # Inspect Blackboard
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
            continue

        # Substrate Audit
        elif cmd == "/audit":
            audit = completer.lm.cortex.audit_synaptic_health()
            print(f"\n{CYAN}--- Physical Synaptic Substrate Audit ---{RESET}")
            for k, v in audit.items():
                print(f"  {k:30s}: {BOLD}{v}{RESET}")
            print(f"{CYAN}-----------------------------------------{RESET}\n")
            continue

        # Full Autonomous Multi-Agent Swarm / Natural Dialogue / Conceptual Q&A
        else:
            t0 = time.perf_counter()
            with swarm_progress("Architect", "Analyzing user intent...") as progress_fn:
                res = completer.execute_task(
                    goal=user_input,
                    filename="solution.py",
                    verify_executable=True,
                    progress_callback=progress_fn,
                )
            dt = time.perf_counter() - t0

            # 1. Conceptual / Explanatory Inquiry (Bypassed swarm, clean Markdown)
            if res.status == "conceptual":
                explanation = res.audit_summary.get("conceptual_response", "")
                print()
                render_markdown(explanation)
                print()
                continue

            # 2. Conversational / Social Dialogue
            if res.status == "conversational":
                reply = res.audit_summary.get("conversational_response", "How can I help you code today?")
                print(f"\n{GREEN}CortexCode>{RESET} {reply}\n")
                continue

            # 3. Actionable Multi-Agent Coding Workflow Completed
            print(f"\n{GREEN}✓{RESET} {BOLD}Task Completed Successfully.{RESET} {DIM}({dt*1000:.1f}ms | Sandbox Verified: {res.sandbox_test_passed}){RESET}")
            if res.git_status and res.git_status.get("is_git_repo"):
                print(f"  {DIM}Git: {res.git_status.get('branch')} ({res.git_status.get('commit_hash')}){RESET}")
            
            if res.patch_diff:
                print(f"\n{CYAN}--- UNIFIED PATCH DIFF ({res.artifacts[0].filename if res.artifacts else 'code'}) ---{RESET}")
                for dline in res.patch_diff.splitlines()[:25]:
                    if dline.startswith("+"):
                        print(f"{GREEN}{dline}{RESET}")
                    elif dline.startswith("-"):
                        print(f"{RED}{dline}{RESET}")
                    elif dline.startswith("@"):
                        print(f"{CYAN}{dline}{RESET}")
                    else:
                        print(f" {dline}")
                if len(res.patch_diff.splitlines()) > 25:
                    print(f"  {DIM}... ({len(res.patch_diff.splitlines()) - 25} more lines in patch diff) ...{RESET}")
                print(f"{CYAN}-------------------------------------------------------{RESET}")

            if res.artifacts:
                art = res.artifacts[0]
                print(f"\n{BOLD}=== GENERATED CODE: {art.filename} ==={RESET}")
                print(art.code)
                print(f"{BOLD}========================================{RESET}\n")

                try:
                    prompt_write = input(f"Save code to '{art.filename}' on disk? [y/N]: ").strip().lower()
                    if prompt_write in ("y", "yes"):
                        with open(art.filename, "w", encoding="utf-8") as f:
                            f.write(art.code)
                        print(f"{GREEN}✓ Successfully saved code to {art.filename}{RESET}\n")
                    else:
                        print(f"{YELLOW}Skipped writing file to disk.{RESET}\n")
                except (KeyboardInterrupt, EOFError):
                    print(f"\n{YELLOW}Skipped writing file.{RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="CortexCode (CORTEX-Coder) Industrial CLI")
    parser.add_argument("--goal", type=str, help="Coding goal to complete in one-shot mode")
    parser.add_argument("--output", type=str, default="solution.py", help="Output filename for code")
    parser.add_argument("--yes", "-y", action="store_true", help="Automatically accept and write patch to disk without prompt")
    args = parser.parse_args()

    completer = MultiAgentTaskCompleter()

    if args.goal:
        with swarm_progress("Architect", "Analyzing goal...") as progress_fn:
            res = completer.execute_task(
                goal=args.goal,
                filename=args.output,
                verify_executable=True,
                progress_callback=progress_fn,
            )
        if res.status == "conceptual":
            render_markdown(res.audit_summary.get("conceptual_response", ""))
        elif res.status == "conversational":
            print(res.audit_summary.get("conversational_response", ""))
        else:
            print(f"[CortexCode] Completed '{args.goal}' in {res.execution_time_sec*1000:.1f}ms")
            if res.patch_diff:
                print("\n--- UNIFIED DIFF ---")
                print(res.patch_diff[:500] + ("..." if len(res.patch_diff) > 500 else ""))
            if res.artifacts:
                should_save = args.yes
                if not should_save:
                    try:
                        ans = input(f"Apply patch to '{args.output}'? [y/N]: ").strip().lower()
                        should_save = ans in ("y", "yes")
                    except (KeyboardInterrupt, EOFError):
                        should_save = False

                if should_save:
                    with open(args.output, "w", encoding="utf-8") as f:
                        f.write(res.artifacts[0].code)
                    print(f"Saved generated code to: {args.output}")
                else:
                    print("Code previewed without saving to disk.")
                print("-" * 60)
                print(res.artifacts[0].code)
                print("-" * 60)
    else:
        run_interactive(completer)


if __name__ == "__main__":
    main()
