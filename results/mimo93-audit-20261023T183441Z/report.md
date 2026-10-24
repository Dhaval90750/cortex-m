# M2M-93 archived result arithmetic audit

- Audit type: read-only arithmetic verification; not an experiment reproduction
- Raw observations: 72 across seeds [111, 222, 333]; mechanisms 24
- Per-row derived field checks: 1512; arithmetic verification: PASS
- This verifies saved arithmetic only; it is not a model rerun or independent scientific replication.

## Recomputed condition summaries

| Condition | n | Raw mean | Saved mean | Pooled IID 95% half-width | Saved half-width | Match |
|---|---:|---:|---:|---:|---:|---|
| clean | 72 | 1.276271595 | 1.276271595 | 0.057831895 | 0.057831895 | PASS |
| rank_1 | 72 | 1.289558759 | 1.289558759 | 0.058061220 | 0.058061220 | PASS |
| rank_2 | 72 | 1.281947044 | 1.281947044 | 0.057269972 | 0.057269972 | PASS |
| rank_3 | 72 | 1.284348314 | 1.284348314 | 0.057044974 | 0.057044974 | PASS |
| rank_4 | 72 | 1.284108409 | 1.284108409 | 0.056388521 | 0.056388521 | PASS |
| rank_8 | 72 | 1.286113145 | 1.286113145 | 0.056469445 | 0.056469445 | PASS |
| rank_16 | 72 | 1.313406015 | 1.313406015 | 0.058620412 | 0.058620412 | PASS |
| random | 72 | 1.290335916 | 1.290335916 | 0.055936566 | 0.055936566 | PASS |

## Interpretation and limitation

The archived script computes 1.96 * sample_sd / sqrt(72) over pooled rows. Rows are nested within three trained seeds (24 mechanisms per seed), so this is a pooled-IID interval, not a seed-clustered uncertainty estimate.
The independent model-training replicates are only three seeds. The pooled interval does not account for within-seed dependence; future confirmatory work should report seed-level paired effects and an appropriate hierarchical/cluster uncertainty analysis.

Raw source files were read without modification. No model was trained in this audit.
