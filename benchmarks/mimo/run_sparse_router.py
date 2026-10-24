"""Evaluate support-conditioned sparse routing on a held-out relational task."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from cortex_m.mimo.sparse_router import SparseEvidenceRouter, candidate_routes, evaluate_routes


ROOT = Path(__file__).resolve().parents[2]
N_INPUTS = 8
N_OUTPUTS = 4
SEEDS = (101, 202, 303)
SCENARIOS = ("normal", "permuted", "uniform_shifted")


def make_episode(seed: int, episode_id: int, scenario: str, support_n: int, query_n: int,
                 noise_std: float = 0.03) -> dict:
    py_rng = random.Random(seed * 100_003 + episode_id)
    gen = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + episode_id)
    routes = candidate_routes(N_INPUTS)
    labels = [py_rng.randrange(len(routes)) for _ in range(N_OUTPUTS)]
    permutation = torch.randperm(N_INPUTS, generator=gen) if scenario == "permuted" else torch.arange(N_INPUTS)

    def sample(n: int, noisy: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if scenario == "uniform_shifted":
            base_x = torch.rand((n, N_INPUTS), generator=gen) * 4.0 - 1.0
        else:
            base_x = torch.randn((n, N_INPUTS), generator=gen)
        base_expert_values = evaluate_routes(base_x, routes)
        y = base_expert_values[:, labels]
        if noisy and noise_std:
            y = y + noise_std * torch.randn(y.shape, generator=gen)
        # Targets are computed in original semantic coordinates, then x is
        # permuted. This makes the scenario a true equivariance test.
        x = base_x.index_select(1, permutation)
        return x, y, evaluate_routes(x, routes)

    sx, sy, _ = sample(support_n, noisy=True)
    qx, qy, _ = sample(query_n, noisy=False)
    # Permutation changes physical indices. Transform labels for evaluation only.
    if scenario == "permuted":
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(N_INPUTS)
        permuted_labels = []
        for label in labels:
            route = routes[label]
            new_a, new_b = int(inverse[route.left]), int(inverse[route.right])
            match = next(i for i, candidate in enumerate(routes)
                         if candidate.operation == route.operation
                         and ((candidate.left, candidate.right) == (new_a, new_b)
                              or (candidate.operation in ("sum", "product")
                                  and (candidate.left, candidate.right) == (new_b, new_a))))
            permuted_labels.append(match)
        labels = permuted_labels
    return {"support_x": sx, "support_y": sy, "query_x": qx, "query_y": qy,
            "route_ids": torch.tensor(labels, dtype=torch.long)}


def evaluate_episode(episode: dict, router: SparseEvidenceRouter, seed: int, episode_id: int) -> dict:
    sx, sy = episode["support_x"], episode["support_y"]
    qx, qy = episode["query_x"], episode["query_y"]
    route_set = candidate_routes(N_INPUTS)
    routed = router.route(sx, sy, qx)
    labels = episode["route_ids"]
    values = evaluate_routes(qx, route_set)

    # Baselines and counterfactuals all use the same task and query rows.
    zero = torch.zeros_like(qy)
    uniform = torch.stack([values.mean(dim=-1)] * N_OUTPUTS, dim=-1)
    gen = torch.Generator(device="cpu").manual_seed(seed * 97_409 + episode_id)
    random_ids = torch.randint(len(route_set), (N_OUTPUTS,), generator=gen)
    random_pred = torch.stack([values[:, random_ids[o]] for o in range(N_OUTPUTS)], dim=-1)
    oracle = torch.stack([values[:, labels[o]] for o in range(N_OUTPUTS)], dim=-1)
    worst_ids = routed.normalized_losses.argmax(dim=-1)
    worst = torch.stack([values[:, worst_ids[o]] for o in range(N_OUTPUTS)], dim=-1)

    mse = lambda pred: float((pred - qy).square().mean().item())
    top_ids = routed.route_indices.tolist()
    per_output_mimo = ((routed.prediction - qy).square().mean(dim=0)).tolist()
    return {
        "mimo_mse": mse(routed.prediction), "oracle_mse": mse(oracle),
        "zero_mse": mse(zero), "uniform_mse": mse(uniform),
        "random_mse": mse(random_pred), "worst_route_mse": mse(worst),
        "route_accuracy": sum(int(a == b) for a, b in zip(top_ids, labels.tolist())) / N_OUTPUTS,
        "exact_task_routes": int(torch.equal(routed.route_indices, labels)),
        "routes_total": N_OUTPUTS, "mean_active_routes_per_output": float((routed.route_weights > 1e-7).sum(dim=-1).float().mean().item()),
        "mean_route_margin": float(routed.confidence_margin.mean().item()),
        "per_output_mimo_mse": per_output_mimo,
        "labels": labels.tolist(), "selected": top_ids, "worst_selected": worst_ids.tolist(),
    }


def summarize(rows: list[dict]) -> dict:
    keys = ("mimo_mse", "oracle_mse", "zero_mse", "uniform_mse", "random_mse",
            "worst_route_mse", "route_accuracy", "exact_task_routes",
            "mean_active_routes_per_output", "mean_route_margin")
    return {key: sum(float(r[key]) for r in rows) / len(rows) for key in keys}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-per-seed", type=int, default=64)
    parser.add_argument("--support", type=int, default=16)
    parser.add_argument("--queries", type=int, default=128)
    parser.add_argument("--noise", type=float, default=0.03)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes_per_seed < 1 or args.support < 2 or args.queries < 1 or args.noise < 0:
        parser.error("episodes, queries, and support must be positive; support >= 2; noise >= 0")
    torch.set_num_threads(1)
    router = SparseEvidenceRouter(args.temperature)
    rows, summaries = [], {}
    started = time.perf_counter()
    for scenario in SCENARIOS:
        scenario_rows = []
        for seed in SEEDS:
            for episode_id in range(args.episodes_per_seed):
                ep = make_episode(seed, episode_id, scenario, args.support, args.queries, args.noise)
                result = evaluate_episode(ep, router, seed, episode_id)
                row = {"scenario": scenario, "seed": seed, "episode": episode_id,
                       "support_rows": args.support, "query_rows": args.queries, **result}
                rows.append(row)
                scenario_rows.append(row)
        summaries[scenario] = summarize(scenario_rows)

    all_rows = rows
    overall = summarize(all_rows)
    # A predeclared diagnostic gate: exact rules should be recovered from the
    # in-distribution support set, and route intervention must hurt vs selection.
    gate = {}
    for scenario, metrics in summaries.items():
        gate[f"{scenario}_route_accuracy_at_least_0_95"] = metrics["route_accuracy"] >= 0.95
        gate[f"{scenario}_mimo_mse_below_zero_baseline"] = metrics["mimo_mse"] < metrics["zero_mse"]
        gate[f"{scenario}_mimo_better_than_random_routes"] = metrics["mimo_mse"] < metrics["random_mse"]
        gate[f"{scenario}_worst_route_hurts_vs_mimo"] = metrics["worst_route_mse"] > metrics["mimo_mse"]
        gate[f"{scenario}_mimo_within_1e_3_of_oracle"] = metrics["mimo_mse"] <= metrics["oracle_mse"] + 1e-3
    in_dist = summaries["normal"]
    passed = all(gate.values())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.output or ROOT / "results" / f"mimo-sparse-router-{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    code_path = Path(__file__).resolve()
    environment = {"python": sys.version, "torch": torch.__version__, "numpy": __import__("numpy").__version__,
                   "pandas": __import__("pandas").__version__, "device": "cpu", "cuda_available": torch.cuda.is_available()}
    report = {"experiment": "MIMO evidence-conditioned sparse relational routing v1",
              "status": "PASS" if passed else "FAIL", "claim_scope": "synthetic explicit relational rule library; not general language reasoning",
              "method_note": "Non-neural task-adaptation baseline: route scores are support-example normalized MSE; Sparsemax supplies sparse gates. No gradient training or language model is used.",
              "config": vars(args) | {"output": str(out), "seeds": list(SEEDS), "scenarios": list(SCENARIOS),
                                      "n_inputs": N_INPUTS, "n_outputs": N_OUTPUTS, "candidate_routes_per_output": len(candidate_routes(N_INPUTS)),
                                      "seconds": time.perf_counter() - started},
              "environment": environment,
              "code_sha256": sha256(code_path), "router_sha256": sha256(ROOT / "src" / "cortex_m" / "mimo" / "sparse_router.py"),
              "per_scenario": summaries, "overall": overall, "gates": gate}
    with (out / "cases.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", *overall.keys()])
        writer.writeheader()
        for name, metrics in summaries.items():
            writer.writerow({"scenario": name, **metrics})
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [f"# MIMO sparse-router v1: {report['status']}", "",
             "Synthetic task-specific evidence routing; this does not test general language ability.", "",
             "| Scenario | Route accuracy | Exact task-route rate | MIMO MSE | Oracle MSE | Zero MSE | Random MSE | Worst-route MSE | Active experts/output |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in summaries.items():
        lines.append(f"| {name} | {m['route_accuracy']:.3f} | {m['exact_task_routes']:.3f} | {m['mimo_mse']:.6f} | {m['oracle_mse']:.6f} | {m['zero_mse']:.6f} | {m['random_mse']:.6f} | {m['worst_route_mse']:.6f} | {m['mean_active_routes_per_output']:.2f} |")
    lines += ["", "## Predeclared gates (required in every scenario)", ""]
    lines += [f"- {'PASS' if ok else 'FAIL'} — {key.replace('_', ' ')}" for key, ok in gate.items()]
    lines += ["", "The learned/adapted gate uses only the supplied support x/y values. Route labels are used only for evaluation. `worst_route_mse` deliberately forces each output through its highest-support-loss expert and is a causal intervention control.",
              "", "This is a transparent, non-neural sparse-routing baseline rather than a trained MIMO network. Its role is to validate the task, support interface, sparse gate, and causal metric before a learned scorer is built.",
              "", f"Runtime: {report['config']['seconds']:.3f} s. Detailed per-episode outcomes are in `cases.jsonl`; configuration and hashes are in `report.json`."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    print(f"Status: {report['status']}; gates={sum(gate.values())}/{len(gate)}; normal route accuracy={in_dist['route_accuracy']:.3f}; MIMO MSE={in_dist['mimo_mse']:.6f}; oracle MSE={in_dist['oracle_mse']:.6f}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
