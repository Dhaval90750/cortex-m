# Model-free rule benchmark

All conditions use deterministic phrase rules and fixed response templates. Facts are extracted from the conversation text; annotations are used only for scoring.

```json
{
  "split": "dev",
  "cases": 80,
  "systems": {
    "current_turn": {
      "n": 80,
      "exact_task_success": 0.25,
      "unsupported_answer_rate": 0,
      "mean_context_words_approx": 11.5,
      "p95_latency_seconds": 0.00039819999074097723,
      "by_category": {
        "conflict": 0,
        "correction": 0,
        "recall": 0,
        "unknown": 1
      }
    },
    "full_transcript": {
      "n": 80,
      "exact_task_success": 1,
      "unsupported_answer_rate": 0,
      "mean_context_words_approx": 61.65,
      "p95_latency_seconds": 0.0003440000000409782,
      "by_category": {
        "conflict": 1,
        "correction": 1,
        "recall": 1,
        "unknown": 1
      }
    },
    "gist_only": {
      "n": 80,
      "exact_task_success": 0.75,
      "unsupported_answer_rate": 0.25,
      "mean_context_words_approx": 7.975,
      "p95_latency_seconds": 0.00034619998768903315,
      "by_category": {
        "conflict": 0,
        "correction": 1,
        "recall": 1,
        "unknown": 1
      }
    },
    "cortex_m": {
      "n": 80,
      "exact_task_success": 1,
      "unsupported_answer_rate": 0,
      "mean_context_words_approx": 16.5125,
      "p95_latency_seconds": 0.0008544000011170283,
      "by_category": {
        "conflict": 1,
        "correction": 1,
        "recall": 1,
        "unknown": 1
      }
    }
  },
  "cortex_context_word_reduction_vs_full": 0.7321573398215734,
  "paired_bootstrap_cortex_vs_full": {
    "cortex_minus_full_transcript": 0,
    "ci95_percentile": [
      0,
      0
    ],
    "pairs": 80,
    "bootstrap_repetitions": 10000,
    "seed": 20261023
  },
  "method_note": "All four systems use explicit vocabulary/phrase rules and fixed response templates. No AI model, API, network, or case annotations are used for extraction, retrieval, or response generation.",
  "run_id": "rule-20261023T174336569642Z",
  "dataset_manifest": "data/generated_v2/manifest.json"
}
```
