"""Evaluation script to test human-friendliness, dynamism, and resilience of CortexLM and CortexCode.

Evaluates outputs for:
1. Natural human cadence (no robotic quotes, no def how(**kwargs) dummy templates).
2. Conversational greetings and check-ins ("hi", "how are you", "who are you").
3. Personal context memory ("my name is Dhaval" -> "what is my name?").
4. Incomplete prompts ("my name is", "you are").
5. Real dynamic code generation (LRU cache, Quicksort, Rate Limiter, Binary Search, Trie).
6. AST syntax validity and runtime sandbox execution.
"""

from __future__ import annotations

import sys
import time
sys.path.insert(0, "src")

from cortex_m import CortexLM, CortexCode, MultiAgentTaskCompleter


def evaluate_human_friendliness():
    print("=" * 80)
    print("  EVALUATING CORTEX-M HUMAN-FRIENDLINESS & DYNAMIC RESILIENCE")
    print("=" * 80)

    # 1. Initialize CortexLM and CortexCode
    print("\n[Step 1] Initializing CortexLM and CortexCode substrates...")
    t0 = time.perf_counter()
    lm = CortexLM(dim=64, gist_dim=32, seed=42, imprint_foundation=True)
    completer = MultiAgentTaskCompleter(lm=lm)
    print(f"  Initialized in {(time.perf_counter() - t0)*1000:.1f}ms\n")

    # -------------------------------------------------------------------------
    # TEST SUITE A: CortexLM Human Conversational Queries
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("  SUITE A: CortexLM Basic Human Interactions")
    print("=" * 80)

    prompts_a = [
        ("hi", ["hello", "connect"]),
        ("how are you", ["operating", "performance", "synaptic"]),
        ("who are you", ["cortexlm", "cortex-m", "synaptic"]),
        ("what can you do", ["capabilities", "coding", "synaptic"]),
        ("my name is", ["let me know your name", "address you"]),
        ("you are", ["cortexlm", "powered by cortex-m"]),
        ("my name is Dhaval", ["hello dhaval", "session memory"]),
        ("what is my name?", ["dhaval"]),
    ]

    for prompt, expected_keywords in prompts_a:
        out = lm.chat(prompt)
        text = out.text
        lower = text.lower()
        passed = all(kw in lower for kw in expected_keywords)
        status = "[PASS]" if passed else "[FAIL]"
        print(f"\nUser > {prompt}")
        print(f"CortexLM > {text}")
        print(f"  Evaluation: {status} | Latency: {out.latency_ms:.1f}ms | Risk: {out.hallucination_risk:.3f}")
        assert passed, f"Failed keyword check for '{prompt}': expected {expected_keywords} in '{text}'"

    # -------------------------------------------------------------------------
    # TEST SUITE B: CortexLM Technical & Systems Knowledge (Clean output)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  SUITE B: CortexLM Technical Human Explanations (No Robotic Badges)")
    print("=" * 80)

    prompts_b = [
        ("What is the difference between Docker and Virtual Machines?", ["kernel", "hypervisor"]),
        ("How do I debug memory leaks in production?", ["profiler", "heap"]),
        ("What is first principles thinking?", ["axiomatic"]),
    ]

    for prompt, expected_keywords in prompts_b:
        out = lm.generate(prompt)
        text = out.text
        lower = text.lower()
        passed = all(kw in lower for kw in expected_keywords)
        # Ensure no robotic badges like "(Remembered from..."
        no_badge = "(remembered from" not in lower
        status = "[PASS]" if (passed and no_badge) else "[FAIL]"
        print(f"\nUser > {prompt}")
        print(f"CortexLM > {text}")
        print(f"  Evaluation: {status} | No Robotic Badge: {no_badge} | Latency: {out.latency_ms:.1f}ms")
        assert passed and no_badge, f"Failed technical check for '{prompt}'"

    # -------------------------------------------------------------------------
    # TEST SUITE C: CortexCode Dynamic Conversational & Coding Swarm
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  SUITE C: CortexCode CLI & Multi-Agent Swarm (Zero Hardcoded Stubs)")
    print("=" * 80)

    # C.1 Conversational input to CortexCode
    print("\n[C.1] Conversational input to CortexCode >> 'how are you'")
    res_conv = completer.execute_task("how are you")
    print(f"  Status: {res_conv.status}")
    print(f"  Response: {res_conv.audit_summary.get('conversational_response')}")
    assert res_conv.status == "conversational"
    assert len(res_conv.artifacts) == 0, "Conversational input must NOT generate fake code files!"
    print("  Evaluation: [PASS] Responded conversationally without fake Python stubs.")

    # C.2 Real coding task: LRU Cache
    print("\n[C.2] Coding task >> 'Implement an LRU cache with capacity limit'")
    res_lru = completer.execute_task("Implement an LRU cache with capacity limit", filename="lru.py")
    art_lru = res_lru.artifacts[0]
    print(f"  Status: {res_lru.status} | Functions: {art_lru.functions} | Classes: {art_lru.classes}")
    assert "LRUCache" in art_lru.classes
    assert res_lru.audit_summary.get("runtime_verified") is True
    print("  Evaluation: [PASS] LRUCache generated with AST validation & sandbox execution.")

    # C.3 Real coding task: Quicksort
    print("\n[C.3] Coding task >> 'Write a quicksort algorithm in python'")
    res_qs = completer.execute_task("Write a quicksort algorithm in python", filename="quicksort.py")
    art_qs = res_qs.artifacts[0]
    print(f"  Status: {res_qs.status} | Functions: {art_qs.functions}")
    assert "quicksort" in art_qs.functions
    assert res_qs.audit_summary.get("runtime_verified") is True
    print("  Evaluation: [PASS] Quicksort generated with AST validation & sandbox execution.")

    # C.4 Real coding task: Rate Limiter
    print("\n[C.4] Coding task >> 'Implement a thread-safe token bucket rate limiter'")
    res_rl = completer.execute_task("Implement a thread-safe token bucket rate limiter", filename="rate_limiter.py")
    art_rl = res_rl.artifacts[0]
    print(f"  Status: {res_rl.status} | Classes: {art_rl.classes}")
    assert "TokenBucketRateLimiter" in art_rl.classes
    assert res_rl.audit_summary.get("runtime_verified") is True
    print("  Evaluation: [PASS] TokenBucketRateLimiter generated with thread locks & AST validation.")

    # C.5 Real coding task: Binary Search
    print("\n[C.5] Coding task >> 'Write a binary search algorithm in python'")
    res_bs = completer.execute_task("Write a binary search algorithm in python", filename="binary_search.py")
    art_bs = res_bs.artifacts[0]
    print(f"  Status: {res_bs.status} | Functions: {art_bs.functions}")
    assert "binary_search" in art_bs.functions
    assert res_bs.audit_summary.get("runtime_verified") is True
    print("  Evaluation: [PASS] Binary search generated with AST validation.")

    # C.6 Real coding task: Trie Prefix Tree
    print("\n[C.6] Coding task >> 'Create a prefix tree trie with autocomplete'")
    res_trie = completer.execute_task("Create a prefix tree trie with autocomplete", filename="trie.py")
    art_trie = res_trie.artifacts[0]
    print(f"  Status: {res_trie.status} | Classes: {art_trie.classes}")
    assert "Trie" in art_trie.classes
    assert res_trie.audit_summary.get("runtime_verified") is True
    print("  Evaluation: [PASS] Trie generated with AST validation.")

    print("\n" + "=" * 80)
    print("  ALL HUMAN-FRIENDLINESS & RESILIENCE TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_human_friendliness()
