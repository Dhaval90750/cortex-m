"""Read-only audit of archived M2M-93 route-swap raw rows and summary.

This verifies arithmetic, not the validity or successful reproduction of the
archived experiment. It does not import PyTorch or modify the archive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


CONDITIONS = ("clean", "rank_1", "rank_2", "rank_3", "rank_4", "rank_8", "rank_16", "random")
ARCHIVE = Path(r"E:\Research\Old MIMO\workspace_snapshot_2026-10-23")
RAW_REL = Path("M2M-93-results/M2M-93-counterfactual-route-all-results.csv")
SUMMARY_REL = Path("M2M-93-results/M2M-93-counterfactual-route-summary.csv")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1)) if n > 1 else 0.0
    half_width = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
    return mean, sd, half_width


def audit(archive: Path) -> dict:
    raw_path, summary_path = archive / RAW_REL, archive / SUMMARY_REL
    raw, saved = read_csv(raw_path), read_csv(summary_path)
    if not raw:
        raise ValueError(f"No raw rows in {raw_path}")
    seeds = sorted({row["Seed"] for row in raw})
    mechanisms = sorted({row["Mechanism"] for row in raw}, key=int)
    bad_derived = []
    derived_checks = 0
    for i, row in enumerate(raw, start=2):
        clean = float(row["clean_MSE"])
        for condition in CONDITIONS[1:]:
            value = float(row[f"{condition}_MSE"])
            checks = {
                "delta": (float(row[f"{condition}_MSE_delta"]), value - clean),
                "ratio": (float(row[f"{condition}_MSE_ratio"]), value / max(clean, 1e-12)),
                "degradation_percent": (
                    float(row[f"{condition}_MSE_degradation_percent"]),
                    (value / max(clean, 1e-12) - 1.0) * 100.0,
                ),
            }
            for name, (actual, expected) in checks.items():
                derived_checks += 1
                if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-7):
                    bad_derived.append({"csv_line": i, "condition": condition, "field": name,
                                        "saved": actual, "recomputed": expected})

    saved_by_condition = {row["Condition"]: row for row in saved}
    results = []
    for condition in CONDITIONS:
        values = [float(row[f"{condition}_MSE"]) for row in raw]
        mean, sd, ci = stats(values)
        ref = saved_by_condition[condition]
        results.append({
            "condition": condition,
            "n_rows": len(values),
            "raw_mean": mean,
            "saved_mean": float(ref["MSE"]),
            "mean_matches": math.isclose(mean, float(ref["MSE"]), rel_tol=1e-9, abs_tol=1e-9),
            "sample_sd": sd,
            "recomputed_pooled_iid_95_half_width": ci,
            "saved_95_half_width": float(ref["MSE_CI95"]),
            "ci_matches_script_formula": math.isclose(ci, float(ref["MSE_CI95"]), rel_tol=1e-8, abs_tol=1e-9),
        })

    seed_means = []
    for seed in seeds:
        group = [row for row in raw if row["Seed"] == seed]
        seed_means.append({
            "seed": int(seed),
            "n_rows": len(group),
            "clean_mean": sum(float(r["clean_MSE"]) for r in group) / len(group),
            **{f"{c}_mean": sum(float(r[f"{c}_MSE"]) for r in group) / len(group)
               for c in CONDITIONS[1:]},
        })
    return {
        "audit_type": "read-only arithmetic verification; not an experiment reproduction",
        "archive_root": str(archive),
        "source_script": str(archive / "m2m_93_counterfactual_route_swaps.py"),
        "raw_csv": str(raw_path),
        "summary_csv": str(summary_path),
        "sha256": {
            "source_script": sha256(archive / "m2m_93_counterfactual_route_swaps.py"),
            "raw_csv": sha256(raw_path),
            "summary_csv": sha256(summary_path),
        },
        "raw_rows": len(raw),
        "seed_ids": [int(s) for s in seeds],
        "mechanism_ids": [int(m) for m in mechanisms],
        "rows_per_seed": {s: sum(row["Seed"] == s for row in raw) for s in seeds},
        "derived_field_checks": derived_checks,
        "derived_column_mismatches": bad_derived,
        "conditions": results,
        "per_seed_means": seed_means,
        "interval_note": (
            "The archived script computes 1.96 * sample_sd / sqrt(72) over pooled rows. "
            "Rows are nested within three trained seeds (24 mechanisms per seed), so this "
            "is a pooled-IID interval, not a seed-clustered uncertainty estimate."
        ),
        "arithmetic_pass": not bad_derived and all(x["mean_matches"] and x["ci_matches_script_formula"] for x in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE, help="Read-only archive snapshot root")
    parser.add_argument("--output", type=Path, help="Optional output directory; defaults to timestamped results folder")
    args = parser.parse_args()
    report = audit(args.archive)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.output or Path(__file__).resolve().parents[1] / "results" / f"mimo93-audit-{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# M2M-93 archived result arithmetic audit", "",
        f"- Audit type: {report['audit_type']}",
        f"- Raw observations: {report['raw_rows']} across seeds {report['seed_ids']}; mechanisms {len(report['mechanism_ids'])}",
        f"- Per-row derived field checks: {report['derived_field_checks']}; arithmetic verification: {'PASS' if report['arithmetic_pass'] else 'FAIL'}",
        "- This verifies saved arithmetic only; it is not a model rerun or independent scientific replication.",
        "", "## Recomputed condition summaries", "",
        "| Condition | n | Raw mean | Saved mean | Pooled IID 95% half-width | Saved half-width | Match |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["conditions"]:
        ok = row["mean_matches"] and row["ci_matches_script_formula"]
        lines.append(f"| {row['condition']} | {row['n_rows']} | {row['raw_mean']:.9f} | {row['saved_mean']:.9f} | {row['recomputed_pooled_iid_95_half_width']:.9f} | {row['saved_95_half_width']:.9f} | {'PASS' if ok else 'FAIL'} |")
    lines += ["", "## Interpretation and limitation", "", report["interval_note"],
              "The independent model-training replicates are only three seeds. The pooled interval does not account for within-seed dependence; future confirmatory work should report seed-level paired effects and an appropriate hierarchical/cluster uncertainty analysis.",
              "", "Raw source files were read without modification. No model was trained in this audit."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    print(f"Arithmetic verification: {'PASS' if report['arithmetic_pass'] else 'FAIL'}; rows={report['raw_rows']}; seeds={len(report['seed_ids'])}")
    return 0 if report["arithmetic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
