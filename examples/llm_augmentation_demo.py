"""End-to-End LLM Augmentation Demonstration using Cortex-M.

Demonstrates how to directly integrate Cortex-M with modern LLMs to:
1. Replace Vector DBs / RAG with Continuous Hopfield Synaptic Memory.
2. Enforce Ground-Truth Provenance & Pathway Resonance Explainability.
3. Predict & Neutralize Hallucinations before Token Generation.
4. Route Prompts via 128-D Factored Gist to Specialized Model Tiers.
5. Host Multi-Tenant Isolated Memory in 2 KB RAM per Tenant.
"""

from __future__ import annotations

import json
from cortex_m import CortexEngine, CortexLLMBridge


def simulate_llm(system_prompt: str, user_prompt: str) -> str:
    """Mock LLM generator illustrating grounded vs ungrounded generation."""
    print("\n--- [LLM Input Prompt] ---")
    print(f"[System]:\n{system_prompt}")
    print(f"[User]:\n{user_prompt}")
    print("--------------------------")

    # In a real system, you would call OpenAI / Anthropic / Gemini / Llama here:
    # response = client.chat.completions.create(model="...", messages=[...])
    if "Port 9443" in system_prompt:
        return "You can reach the secured gRPC streaming service at Port 9443 with mTLS enabled."
    return "Please consult the standard service directory for network connection endpoints."


def main():
    print("=" * 70)
    print("CORTEX-M: TURNKEY LLM AUGMENTATION & COGNITIVE PLATFORM")
    print("=" * 70)

    # 1. Initialize Cortex-M Engine
    engine = CortexEngine()

    # 2. Learn Domain Knowledge Directly into Synaptic Weights W (No Vector DB!)
    print("\n[Step 1] Storing Ground-Truth Domain Memories in Synapses...")
    engine.remember(
        "The primary gRPC streaming service operates on Port 9443 using mTLS encryption.",
        memory_id="grpc_endpoint_spec",
        source="devops_network_manifest_v3",
        salience=1.0,
    )
    engine.remember(
        "Financial data retention threshold is strictly 7 years under regulatory compliance.",
        memory_id="retention_policy",
        source="legal_compliance_handbook",
        salience=1.0,
    )

    # 3. Grounded Associative Recall & Explainability
    print("\n[Step 2] Grounded Associative Recall with Provenance Tracing...")
    query = "What port and encryption does the gRPC service use?"
    recall_res = engine.recall(query)
    print(f"  User Query        : '{query}'")
    print(f"  Recalled Attractor: {recall_res.memory_id} (Similarity: {recall_res.attractor_similarity:.3f})")
    print(f"  Resonance Score   : {recall_res.resonance_score:.3f}")
    print(f"  Provenance Explain: {recall_res.explain()}")

    # 4. LLM Bridge: Prompt Augmentation & Hallucination Defense
    print("\n[Step 3] Augmenting LLM Generation & Defending Against Hallucinations...")
    bridge = engine.get_llm_bridge()
    augmented = bridge.augment_prompt(query, system_prompt="You are a DevOps network engineer.")
    risk = augmented["hallucination_assessment"]

    print(f"  Epistemic Confidence   : {risk['epistemic_confidence'] * 100:.1f}%")
    print(f"  Hallucination Risk Score : {risk['hallucination_risk_score']:.4f}")
    print(f"  Assessment              : {risk['assessment']}")
    print(f"  Context Injected        : {augmented['context_injected']}")
    print(f"  Cortex-M Engine Latency : {augmented['latency_ms']} ms")

    # 5. Executing Wrapped LLM Pipeline
    print("\n[Step 4] Executing LLM with Cortex-M Guardrails...")
    wrapped_generator = bridge.wrap_llm(simulate_llm)
    output = wrapped_generator(query)
    print(f"\n[Generated Output]:\n{output['response']}")

    # 6. Multi-Tenant Ephemeral Paging (Zero Cross-Tenant Leakage)
    print("\n[Step 5] Multi-Tenant Synaptic Delta Isolation (2 KB Footprint)...")
    with engine.tenant("tenant_acme_corp") as acme:
        acme.remember("ACME internal server IP is 10.200.4.15.")
        acme_res = acme.recall("What is ACME server IP?")
        print(f"  Tenant ACME Recall : '{acme_res.text}'")

    with engine.tenant("tenant_globex") as globex:
        # Globex cannot see ACME internal IP
        globex_res = globex.recall("What is ACME server IP?")
        print(f"  Tenant Globex Recall (Isolated): '{globex_res.text}'")

    # 7. Substrate Health Audit
    print("\n[Step 6] Auditing Substrate Synaptic Health...")
    health = engine.audit()
    print(f"  Synaptic Sparsity    : {health['synaptic_sparsity_pct']}%")
    print(f"  Substrate Health     : {health['substrate_health']}")
    print(f"  Memories in Synapses : {health['total_memories_stored']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
