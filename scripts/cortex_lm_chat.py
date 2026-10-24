"""Interactive Terminal Chat & Synaptic Exploration Interface for CortexLM.

Run natively with:
    python scripts/cortex_lm_chat.py

Provides an interactive CLI where users can:
- Converse with CortexLM across systems, mathematics, algorithms, and neural architecture.
- Learn new facts in real-time into synaptic weights (/learn <fact>).
- Selectively erase memories via Anti-Hebbian depression (/forget <id>).
- Inspect Lyapunov energy descent and pathway resonance diagnostics (/prov).
- Run biological sleep consolidation (/sleep) and substrate audits (/audit).
"""

from __future__ import annotations

try:
    import readline  # type: ignore[import-untyped]
except ImportError:
    pass
import sys
import time

# Ensure src is in python path
sys.path.insert(0, "src")

from cortex_m import CortexLM, CortexLMOutput


BANNER = r"""
================================================================================
   ______           __                 __    __  ___
  / ____/___  _____/ /____  _  __     / /   /  |/  /
 / /   / __ \/ ___/ __/ _ \| |/_/____/ /   / /|_/ / 
/ /___/ /_/ / /  / /_/  __/>  </_____/ /___/ /  / /  
\____/\____/_/   \__/\___/_/|_|     /_____/_/  /_/   
  Continuous Attractor Native Language Model | Bio-Inspired Neural Substrate
================================================================================
 * Zero-Database Architecture: Knowledge encoded directly into Synaptic Weights W
 * Lyapunov Energy Guardrails: Mathematical immunity against hallucinations
 * Real-Time Plasticity: Instant Hebbian learning & Anti-Hebbian unlearning (<1ms)
 * Dorsal-Ventral Streams: 128-D Pre-Verbal Gist + Verbatim Numeric/Code Precision
================================================================================
Commands:
  /learn <fact>           -> Teach CortexLM a new fact in real-time into synapses
  /forget <id>            -> Instant targeted unlearning via anti-Hebbian depression
  /audit                  -> View synaptic capacity, density, and saturation
  /sleep                  -> Trigger biological sleep consolidation and pruning
  /prov                   -> Show deep causal provenance & Lyapunov metrics
  /clear                  -> Reset prefrontal working memory context
  /help                   -> Display command reference
  /exit                   -> Quit
================================================================================
"""


def main():
    print(BANNER)
    print("[CortexLM] Initializing synaptic neural substrate...")
    t0 = time.perf_counter()
    lm = CortexLM(dim=64, gist_dim=32, seed=42, imprint_foundation=True)
    init_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[CortexLM] Substrate ready in {init_ms:.1f} ms. Foundational knowledge imprinted.")
    print("Type a prompt or command to begin chatting.\n")

    last_output: CortexLMOutput | None = None

    while True:
        try:
            user_input = input("\033[1;36mUser>\033[0m ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[CortexLM] Exiting. Synaptic state preserved.")
            break

        if not user_input:
            continue

        # Command handling
        if user_input in ("/exit", "/quit", "exit", "quit"):
            print("[CortexLM] Shutting down continuous neural substrate. Goodbye!")
            break

        if user_input == "/help":
            print("""
Available Commands:
  /learn <fact>      : Learn a new fact into synaptic weights in one forward step.
  /forget <id>       : Trigger anti-Hebbian depression to erase an attractor basin.
  /audit             : Inspect physical synaptic matrix health, capacity, and sparsity.
  /sleep             : Replay stored attractors and prune weak connections.
  /prov              : Inspect detailed causal explainability and energy drop of last reply.
  /clear             : Clear prefrontal working memory turn context.
  /help              : Show this message.
  /exit              : Exit chat.
""")
            continue

        if user_input.startswith("/learn "):
            raw_fact = user_input[7:].strip()
            if not raw_fact:
                print("[Error] Specify a fact to learn. Example: /learn The server IP is 192.168.1.50")
                continue
            res = lm.learn(raw_fact, source="interactive_user")
            if res.get("status") == "blocked_by_safety_critic":
                print(f"\033[1;31m[Safety Critic Alert]\033[0m Learning blocked: {res.get('reason')}")
            else:
                mem_id = res.get("memory_id")
                delta_norm = res.get("synaptic_delta_norm", 0.0)
                print(f"\033[1;32m[Hebbian Plasticity]\033[0m Learned into physical synapses! (ID: {mem_id}, ΔW norm: {delta_norm:.4f})")
            continue

        if user_input.startswith("/forget "):
            target = user_input[8:].strip()
            if not target:
                print("[Error] Specify a memory ID or text to forget. Example: /forget secret_pin")
                continue
            res = lm.forget(target)
            if res.get("status") == "unlearned":
                print(f"\033[1;33m[Anti-Hebbian Depression]\033[0m Basin erased! (ID: {target}, Synaptic ΔW: {res.get('synaptic_delta_norm', 0.0):.4f})")
            elif res.get("status") == "blocked_by_safety_critic":
                print(f"\033[1;31m[Safety Critic Alert]\033[0m Unlearning blocked: {res.get('reason')}")
            else:
                print(f"[Notice] Memory '{target}' not found in active attractors.")
            continue

        if user_input == "/audit":
            audit_res = lm.audit()
            print("\n--- Physical Synaptic Substrate Audit ---")
            for k, v in audit_res.items():
                print(f"  {k:30s}: {v}")
            print("-----------------------------------------\n")
            continue

        if user_input == "/sleep":
            print("[CortexLM] Initiating biological sleep consolidation and homeostatic pruning...")
            sleep_res = lm.sleep()
            print(f"\033[1;32m[Sleep Consolidation Complete]\033[0m")
            print(f"  Attractors Replayed: {sleep_res.get('replayed_attractors')}")
            print(f"  Synaptic Sparsity  : {sleep_res.get('synaptic_sparsity_pct', 0.0):.2f}%")
            print(f"  Pruned Synaptic Edges: {sleep_res.get('pruned_synapses')}\n")
            continue

        if user_input == "/clear":
            lm.cortex.working_memory.reset()
            print("[CortexLM] Working memory context reset.")
            continue

        if user_input == "/prov":
            if not last_output:
                print("[Notice] No previous generation to inspect.")
                continue
            print("\n--- Provenance & Causal Explainability ---")
            print(f"  Grounded in Attractor : {last_output.is_grounded}")
            print(f"  Lyapunov Risk Score   : {last_output.hallucination_risk:.4f}")
            print(f"  Inference Latency     : {last_output.latency_ms:.2f} ms")
            print(f"  Synaptic Provenance   : {last_output.provenance}")
            print(f"  Causal Explanation    : {last_output.explainability}")
            print("------------------------------------------\n")
            continue

        # Standard Conversational Generation via Prefrontal Working Memory
        out = lm.chat(user_input)
        last_output = out

        # Display response
        if out.is_grounded:
            prov_id = out.provenance.get("memory_id", "synaptic") if out.provenance else "synaptic"
            res_score = out.provenance.get("resonance_score", 0.70) if out.provenance else 0.70
            print(f"\n\033[1;32mCortexLM>\033[0m {out.text}")
            print(f"\033[2m  [Attractor: {prov_id} | Pathway Resonance: {res_score*100:.0f}% | Latency: {out.latency_ms:.1f}ms | Risk: {out.hallucination_risk:.3f}]\033[0m\n")
        else:
            print(f"\n\033[1;33mCortexLM [Abstention]>\033[0m {out.text}")
            print(f"\033[2m  [Lyapunov Energy Shield: Refused Hallucination | Risk: {out.hallucination_risk:.3f} | Latency: {out.latency_ms:.1f}ms]\033[0m\n")


if __name__ == "__main__":
    main()
