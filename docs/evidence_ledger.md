# Evidence ledger

| Claim | Status | Evidence / limitation |
|---|---|---|
| Sparsemax-K can solve the original hidden-rule N=8 task | Historical, bounded | Old MIMO master record reports 96.20% mean exact graph over ten seeds; synthetic benchmark only. |
| Sparse routing is the causal reason for M2M output quality | Unresolved | M2M-91/92/93 route perturbation effects are often small; MIMO routing targets are questionable. |
| Existing SLM-01 proves support memory helps/hurts an SLM | Invalid benchmark claim | Support labels encode aggregate relation effects while query outputs use per-output relations and separately sampled transforms. |
| Existing LLM integration proves M2M improves transformer reasoning | Not supported | One unseeded tiny sorting run: SwiGLU perplexity 15.62 / 20,178 params / 1.0 s; M2M 18.97 / 10,163 / 15.8 s on CPU. |
| Deterministic gist v0 handles the seven edge cases and 100 generated variants | Unit-level only | Archived prototype passed 107/107 cases built around its own behavior; not an independent language evaluation. |
| Cortex-M deterministic core handles annotated state transitions on the generated split | Component behavior only | Fresh core run: dev 80/80 and holdout 240/240; 16/16 unit and stress tests pass. Tests use gold annotations and measure no language-model extraction/generation or transcript baseline. Latest artifacts are `results/core-20261023T173926067643Z/` (dev) and `results/core-20261023T173926545884Z/` (holdout). |
| Model-free Cortex-M preserves explicit facts on the controlled phrase benchmark | Supported only for v2 synthetic grammar | Rule-based extractor matched annotations on all generated cases. Holdout: current-turn 25%, full transcript 100%, latest-value gist 75% with 25% unsupported answers, Cortex-M 100% with 0% unsupported answers and 72.9% fewer approximate context words than full transcript. Paired success difference vs full transcript 0 pp, bootstrap 95% CI [0, 0], n=240. Cortex-M p95 was 2.16x transcript p95 in a sub-millisecond microbenchmark, so the relative latency target did not pass. Latest output: `results/rule-20261023T174812718078Z/`. No AI/API was used; this does not test open-vocabulary understanding. |
| Existing M2M route evidence clears the Cortex-M integration gate | No; keep M2M separate | Read-only review of archived M2M-91/92/93 summaries. At 100% router-score perturbation in M2M-91, routing top-half overlap fell to 0.493 but MSE improved by 0.176% (reported 95% CI half-width 0.078 pp). M2M-92 hard-top1 and hard-worst increased MSE only 1.04% and 2.91%; zero-top1 increased it 13.07%. M2M-93 rank swaps were mostly below 1% until rank-16 (+2.91%). These saved 72-example results do not meet the planned repeated-seed, matched-control, compositional-gain and causal-route gate; this was a file review, not a fresh rerun. |
| Cortex M improves multi-turn continuity with an LLM | Outside current scope | The user selected a model-free implementation; results above compare deterministic rule systems only. |

Every new experiment should append: claim, hypothesis, code commit, data hash,
model/runtime version, command, metrics, confidence interval, failure modes,
and interpretation. Distinguish direct measurement from interpretation.
