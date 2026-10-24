"""Cognitive Analysis & State Profiling Demo using Cortex-M.

Demonstrates advanced cognitive analysis capabilities:
1. Continuous Lyapunov Energy Relaxation Profiling.
2. 128-D Factored Gist Semantic Intent Deconstruction.
3. Substrate Synaptic Sparsity, Homeostasis, and Sprouting Diagnostics.
4. Epistemic Uncertainty & Epistemic Basin Depth Analysis.
"""

from __future__ import annotations

import json
from cortex_m import CortexEngine, CognitiveAnalyzer


def main():
    print("=" * 70)
    print("CORTEX-M: COGNITIVE ANALYSIS & STATE DIAGNOSTICS SUITE")
    print("=" * 70)

    engine = CortexEngine()

    # 1. Encode Diverse Cognitive Experiences
    print("\n[Analysis 1] Populating Continuous Synaptic Landscape...")
    memories = [
        ("Autonomous agent architecture requires dynamic routing and episodic recall.", "agent_arch", "research_paper"),
        ("Rust memory safety prevents data races and dangling pointer dereferences.", "rust_safety", "compiler_manual"),
        ("Database indexing with B-Trees has logarithmic asymptotic time complexity.", "db_index", "algo_textbook"),
    ]

    for text, mid, src in memories:
        trace = engine.remember(text, memory_id=mid, source=src)
        print(f"  Encoded Attractor '{mid}': Initial Energy = {trace['initial_energy']:.3f} -> Final Energy = {trace['final_energy']:.3f}")

    # 2. 128-D Factored Gist Semantic Deconstruction
    print("\n[Analysis 2] 128-D Factored Gist Intent Deconstruction...")
    prompt = "Synthesize an efficient Rust function to calculate SHA-256 digests under 50 lines."
    dual_out = engine.dual_stream(prompt)
    factored = dual_out["factored_heads"]

    print(f"  Input Prompt: '{prompt}'")
    print("  Partitioned Factored Heads (128-dim Intent Vector):")
    for head_name, vector_sample in factored.items():
        print(f"    - Head [{head_name:<10}]: Sample {vector_sample}")

    dorsal = engine.cortex.dual_stream.dorsal_buffer
    print(f"  Dorsal Syntactic Capture:")
    print(f"    - Numeric Tokens Captured : {dorsal.numeric_tokens}")
    print(f"    - Verbatim Raw Text Length: {len(dorsal.raw_text_history[-1])} chars")

    # 3. Lyapunov Energy Relaxation & Attractor Convergence
    print("\n[Analysis 3] Continuous Lyapunov Energy Trajectory Analysis...")
    query = "Why does Rust guarantee memory safety?"
    recall_res = engine.recall(query)
    raw = recall_res.raw
    trace_summary = raw["trace_summary"]

    print(f"  Query State       : '{query}'")
    print(f"  Attractor Matched : '{recall_res.memory_id}' (Similarity: {recall_res.attractor_similarity:.4f})")
    print(f"  Convergence Steps : {raw['steps_taken']} relaxation iterations")
    print(f"  Lyapunov Energy Drop: {raw['energy_drop']:.4f}")
    print(f"  Pathway Resonance : {recall_res.resonance_score:.4f}")
    print(f"  Explainability    : {recall_res.explain()}")

    # 4. Sleep Consolidation & Synaptic Homeostasis Analysis
    print("\n[Analysis 4] Biological Sleep Replay & Pruning Analysis...")
    pre_health = engine.audit()
    print(f"  Pre-Sleep Sparsity : {pre_health['synaptic_sparsity_pct']}%")

    sleep_report = engine.sleep(replay_iterations=2, noise_level=0.10, decay_factor=0.90)
    print(f"  Synapses Pruned    : {sleep_report['pruned_synapses']} connections")
    print(f"  Sprouted Synapses  : {sleep_report['sprouted_synapses']} reactivated edges")
    print(f"  Post-Sleep Sparsity: {sleep_report['sparsity_pct']}%")

    post_health = engine.audit()
    print(f"  Substrate Health   : {post_health['substrate_health']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
