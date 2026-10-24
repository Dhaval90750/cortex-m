# Cortex-M acceptance report: PARTIAL

Created: 2026-10-23T18:09:43.962002+00:00

## Functional end-to-end cases

| Case | Result | Details |
|---|---|---|
| explicit recall | PASS | Your recorded preferred editor is Vim. |
| correction and superseded history | PASS | Your recorded preferred editor is Emacs. |
| conflict triggers clarification | PASS | I have conflicting details. Which value should I use? |
| uncertain statement is rejected | PASS | I don't have that detail recorded, so I won't guess. |
| unlisted value is rejected | PASS | I don't have that detail recorded, so I won't guess. |
| SQLite persistence after reopen | PASS | See report.json |
| clear removes stored fact | PASS | See report.json |
| open-vocabulary name survives persistence and is recalled | PASS | See report.json |
| unavailable storage falls back safely | PASS | Memory is unavailable. Please repeat that detail. |
| corrupt database fails closed | PASS | See report.json |

## Saved benchmark gates

Source: `E:\Research\Cortex-M\results\rule-20261023T174812718078Z\summary.json`

| Gate | Result | Observed | Target |
|---|---|---:|---:|
| exact success at least full-transcript baseline | PASS | `1` | >= `1` |
| unsupported answer rate <= 2% | PASS | `0` | <= `0.02` |
| approximate context-word reduction >= 30% | PASS | `0.7292753225363711` | >= `0.3` |
| p95 latency ratio vs full transcript <= 2.0 | FAIL | `2.165247993762025` | <= `2.0` |
| recall, correction, conflict, unknown each >= 95% | PASS | `{"conflict": 1, "correction": 1, "recall": 1, "unknown": 1}` | each >= `0.95` |

## Interpretation

Functional scenarios passed, but at least one predefined research gate did not. The implementation is functionally demonstrated on these cases; comparative acceptance is partial.

This check uses deterministic rules and fixed response templates; it does not call an AI model, external API, or network. Context-word reduction is a proxy, not measured model-token savings. The benchmark is a controlled synthetic evaluation, not a general language-understanding result.

## Reproduce

```powershell
.\scripts\python.ps1 scripts\walkthrough.py
.\scripts\python.ps1 scripts\check_success.py
.\scripts\python.ps1 -m unittest discover -s tests -v
```
