# MIMO with retrieved dynamic task gist

## Experiment

A two-rule N=8 relational task assigns each example a rule mode independently of its numeric input. Four outputs each sum two semantic input roles; input nodes are randomly permuted and accompanied by role descriptors. Gist carries the active task mode for the current request.

The paired conditions used identical generated tensors, 4,000 train examples, 1,200 held-out test examples, 500 epochs, batch size 128, Adam at 0.0012, fixed temperature decay, and ten seeds: 101, 202, 303, 404, 505, 606, 707, 808, 909, 1001. The no-gist control is the archived M2M architecture trained on the same random-mode task. The gist model adds a learned encoder of the retrieved two-value mode into the context state used by the router. Evaluation perturbs only that gist: correct, flipped, and zero.

## Results

- Correct gist: mean exact graph 95.83%; nine seeds scored 100%, seed 808 scored 58.33%; mean MSE 0.01365.
- No gist: mean exact graph 0.017%; mean MSE 0.88455.
- Flipped gist: mean exact graph 0%; mean MSE 3.48393.
- Zero gist: mean exact graph 1.36%; mean MSE 2.12699.

This provides evidence that retrieved mode context affects the router and is useful on this bounded task. Seed 808's failure is retained; seed 101 is used in the visualizer because it is the first predeclared seed, not because it was chosen by test score.

## Runtime proof

The one-prompt visualizer retrieves the latest explicit `mode` fact for every `mimo_gist_n8` request. A live run stored alpha, executed the task, corrected the fact to beta, then reran the identical input. Conditioning vectors changed from `[1,0]` to `[0,1]`; both route selection and output changed. Missing/conflicted facts abstain before inference. The test facts were cleared. See `live-e2e.json`.

## Reproduce

From `E:\Research\Cortex-M`:

```powershell
$env:PYTHONPATH = 'E:\Research\Cortex-M\src'
$env:MIMO_GIST_RESULTS_DIR = 'E:\Research\Cortex-M\results\mimo-gist-conditioned'
& 'E:\Research\.venv-mimo\Scripts\python.exe' -u benchmarks/mimo/run_gist_conditioned.py
```

Config, data tensor hashes, code hashes, checkpoint hash, per-seed values, and CSV hash are in `summary.json` and `seed_metrics.csv`. The final harness/model code was also rerun on seed 101; graph accuracy and MSE matched that row exactly.

## Limits and next step

The runtime accepts only the `mode` schema with alpha/beta. This is dynamic retrieval per task within a controlled schema, not open-ended gist extraction. A separate direct-to-route gist architecture scored only 74.95% mean exact graph and was rejected. Next build a variable-size support-set gist from user examples and evaluate on held-out route graphs/support sizes; do not treat the current two-value control as universal.
