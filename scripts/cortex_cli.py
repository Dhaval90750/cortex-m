"""Cortex-M Command Line Interface (CLI).

Provides terminal access to Cortex-M cognitive capabilities:
- learn: Store memories directly in synaptic weights W
- recall: Associative recall with pathway resonance & provenance
- augment: Augment prompts with hallucination risk assessment for LLMs
- analyze: 128-D factored gist and Lyapunov energy state analysis
- audit: Substrate health and capacity metrics
- sleep: Run biological sleep consolidation and synaptic pruning
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch

from cortex_m import CortexEngine


def main():
    parser = argparse.ArgumentParser(description="Cortex-M: Cognitive Neural Platform CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # learn
    learn_p = subparsers.add_parser("learn", help="Store ground-truth memory in synaptic weights")
    learn_p.add_argument("--text", "-t", required=True, help="Text to learn")
    learn_p.add_argument("--id", "-i", default=None, help="Memory identifier")
    learn_p.add_argument("--source", "-s", default="cli_user", help="Source provenance context")
    learn_p.add_argument("--salience", type=float, default=1.0, help="Plasticity salience weight")

    # recall
    recall_p = subparsers.add_parser("recall", help="Associative recall from partial or noisy cue")
    recall_p.add_argument("--query", "-q", required=True, help="Query cue text")

    # augment
    augment_p = subparsers.add_parser("augment", help="Augment LLM prompt with grounded synaptic context & risk audit")
    augment_p.add_argument("--query", "-q", required=True, help="User query")
    augment_p.add_argument("--system", default="You are an AI assistant.", help="Base system prompt")
    augment_p.add_argument("--tenant", default=None, help="Tenant ID for isolated memory")

    # analyze
    analyze_p = subparsers.add_parser("analyze", help="Extract 128-D factored gist and syntactic dorsal tokens")
    analyze_p.add_argument("--prompt", "-p", required=True, help="Input prompt to analyze")

    # audit
    subparsers.add_parser("audit", help="Audit synaptic substrate health, capacity, and sparsity")

    # sleep
    subparsers.add_parser("sleep", help="Run biological sleep consolidation and synaptic pruning")

    args = parser.parse_args()
    engine = CortexEngine()

    def json_serializer(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    if args.command == "learn":
        res = engine.remember(args.text, memory_id=args.id, source=args.source, salience=args.salience)
        print(json.dumps(res, indent=2, default=json_serializer))

    elif args.command == "recall":
        rec = engine.recall(args.query)
        out = {
            "recalled_fact": rec.text,
            "memory_id": rec.memory_id,
            "attractor_similarity": rec.attractor_similarity,
            "resonance_score": rec.resonance_score,
            "explainability": rec.explain(),
        }
        print(json.dumps(out, indent=2, default=json_serializer))

    elif args.command == "augment":
        bridge = engine.get_llm_bridge()
        aug = bridge.augment_prompt(args.query, system_prompt=args.system, tenant_id=args.tenant)
        print(json.dumps(aug, indent=2, default=json_serializer))

    elif args.command == "analyze":
        dual = engine.dual_stream(args.prompt)
        print(json.dumps(dual, indent=2, default=json_serializer))

    elif args.command == "audit":
        health = engine.audit()
        print(json.dumps(health, indent=2, default=json_serializer))

    elif args.command == "sleep":
        report = engine.sleep()
        print(json.dumps(report, indent=2, default=json_serializer))


if __name__ == "__main__":
    main()
