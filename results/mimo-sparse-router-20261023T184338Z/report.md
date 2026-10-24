# MIMO sparse-router v1: PASS

Synthetic task-specific evidence routing; this does not test general language ability.

| Scenario | Route accuracy | Exact task-route rate | MIMO MSE | Oracle MSE | Zero MSE | Random MSE | Worst-route MSE | Active experts/output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 1.000 | 1.000 | 0.000000 | 0.000000 | 1.480991 | 2.724942 | 4.915672 | 1.00 |
| permuted | 1.000 | 1.000 | 0.000000 | 0.000000 | 1.466793 | 2.742394 | 4.928787 | 1.00 |
| uniform_shifted | 1.000 | 1.000 | 0.000000 | 0.000000 | 3.913051 | 5.589894 | 9.130240 | 1.00 |

## Predeclared gates (required in every scenario)

- PASS — normal route accuracy at least 0 95
- PASS — normal mimo mse below zero baseline
- PASS — normal mimo better than random routes
- PASS — normal worst route hurts vs mimo
- PASS — normal mimo within 1e 3 of oracle
- PASS — permuted route accuracy at least 0 95
- PASS — permuted mimo mse below zero baseline
- PASS — permuted mimo better than random routes
- PASS — permuted worst route hurts vs mimo
- PASS — permuted mimo within 1e 3 of oracle
- PASS — uniform shifted route accuracy at least 0 95
- PASS — uniform shifted mimo mse below zero baseline
- PASS — uniform shifted mimo better than random routes
- PASS — uniform shifted worst route hurts vs mimo
- PASS — uniform shifted mimo within 1e 3 of oracle

The learned/adapted gate uses only the supplied support x/y values. Route labels are used only for evaluation. `worst_route_mse` deliberately forces each output through its highest-support-loss expert and is a causal intervention control.

This is a transparent, non-neural sparse-routing baseline rather than a trained MIMO network. Its role is to validate the task, support interface, sparse gate, and causal metric before a learned scorer is built.

Runtime: 2.946 s. Detailed per-episode outcomes are in `cases.jsonl`; configuration and hashes are in `report.json`.
