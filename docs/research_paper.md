# Cortex-M: Model-Free Memory for Explicit User Facts

**Research status:** controlled synthetic evaluation completed; open-vocabulary
and production behavior not established.

## Abstract

Cortex-M is a model-free system for retaining explicitly stated user facts
across multi-turn conversations. It combines a configured phrase lexicon,
SQLite event storage, correction and conflict tracking, lexical retrieval, and
fixed response templates. It loads no AI model and calls no external API. On a
versioned synthetic benchmark of 240 held-out conversations, Cortex-M matched
a full-transcript rule system at 100% exact task success while presenting
72.9% fewer approximate evidence words to its response stage. A latest-value-only gist reached 75% and
produced an unsupported answer in 25% of cases, all associated with conflicting
facts. Cortex-M was slower than the minimal full-transcript baseline in a
sub-millisecond timing test. The results support a narrow claim about explicit
fact memory under the benchmark's fixed vocabulary and phrasing; they do not
show general language understanding or an advantage over AI systems.

## 1. Research question

Can a small deterministic system retain explicit facts, apply corrections,
recognize unresolved conflicts, and abstain on unknown details while reducing
the amount of evidence retained for the response stage?

The user selected a route that does not depend on local AI hardware or hosted
inference. Cortex-M therefore treats language input as a bounded interface:
known keys and values are recognized by rules; unsupported wording is not
guessed.

## 2. System

### 2.1 Extraction

`RuleFactExtractor` uses a configured lexicon and regular expressions. It
recognizes a key/value match and correction cues, and retains the source turn
and exact source text. It rejects phrases marked as unstated and does not
generate facts from an absent value. Unknown values or paraphrases fall outside
its current coverage.

### 2.2 Memory and response

Facts are stored in SQLite. A direct correction supersedes earlier values while
retaining event history. Distinct explicit values without a correction remain
conflicted and cause a clarification request. An unknown value causes an
abstention; unavailable or corrupt storage causes a transparent fallback.
Retrieval prioritizes an exact key match and excludes unrelated conflicts.
Replies are fixed templates grounded in the response plan.

### 2.3 Runtime boundary

The main execution path uses Python standard-library components. It does not
load model weights, use PyTorch, call llama.cpp, use a hosted AI API, or make
network requests. A runnable console demo is available through
`scripts/demo.py`.

## 3. Methods

### 3.1 Dataset

Dataset version 2 was generated with seed `271828`. It contains 80 development
and 240 held-out cases, balanced across four categories: recall, correction,
conflict, and unknown or memory outage. Each case contains eight irrelevant
history turns after the fact statement. Dev and holdout use separate primary
phrasing pools and different key/value pairings. The conflict cases use
explicit statements by the same speaker on both sides. The configured lexicon
has 12 fact keys and 24 values.

The SHA-256 hashes are recorded in `data/generated_v2/manifest.json`:

- Dev: `cc5e5aca91971f147ea6415a75fd80db7bf632b87efa82b7dc2fa1955bfbd689`
- Holdout: `c4e086b6dcc423ceeecd262ac9bfcf8542585fc5d2f8b8df7fa44b565ea456a8`
- Lexicon: `b37db6a593079018d695ab4c6b1416e31c00bfcf53e6339ad2510d493c462d5e`

The benchmark first checks that rule extraction matches the case annotations.
Annotations are used for this corpus check and for outcome scoring, not by any
system to extract, retrieve, or render an answer.

### 3.2 Conditions

All conditions use the same rule extractor, response wording, cases, and
lexicon.

1. **Current turn:** extract only the latest user utterance plus the query.
2. **Full transcript:** re-extract all user utterances at query time and resolve
   corrections and conflicts from the complete event list.
3. **Gist only:** retain the latest value for each key, dropping conflict
   history.
4. **Cortex-M:** store facts and provenance in SQLite, retrieve the queried
   key, and preserve correction/conflict status when planning the reply.

### 3.3 Measures

Exact task success requires both the expected action and (for answers) the
expected value. Unsupported-answer rate counts answers with a value other than
the annotated correct value. Context size is approximated by whitespace word
counts of retained/presented text; it is neither a model-token measurement nor
a measure of all text scanned when facts are first extracted. Per-case elapsed time is
reported with a percentile bootstrap of the paired success difference between
Cortex-M and full transcript (10,000 resamples, seed `20261023`).

## 4. Results

### 4.1 Held-out results

| System | Exact task success | Unsupported answers | Mean approximate context words | p95 elapsed time |
|---|---:|---:|---:|---:|
| Current turn | 25.0% | 0.0% | 11.50 | 0.053 ms |
| Full transcript | 100.0% | 0.0% | 60.72 | 0.341 ms |
| Gist only | 75.0% | 25.0% | 8.00 | 0.323 ms |
| Cortex-M | 100.0% | 0.0% | 16.44 | 0.738 ms |

Cortex-M reduced its approximate evidence-gist footprint by **72.9%** relative
to full transcript, while matching full-transcript task success. This does not
measure AI prompt or inference savings: no language model is used, and Cortex-M
still scans incoming text when extracting facts. The paired success
difference was **0.0 percentage points** (95% percentile bootstrap CI **[0,
0]**, 240 paired cases). The gist-only condition failed all 60 conflict cases
because replacing history with a single latest value removes evidence that the
statements disagree. The current-turn condition retained only 25% success,
mostly by correctly handling unknown/outage cases; it could not recall facts
from earlier turns.

Cortex-M's p95 time was about **2.16×** the full-transcript condition in this
run. Both values were below one millisecond, where timer and process noise are
material. The planned relative latency threshold of 2× was therefore not met.
These numbers describe this Python microbenchmark only.

### 4.2 Development split

On 80 development cases, current-turn success was 25%, full transcript 100%,
gist only 75%, and Cortex-M 100%. Cortex-M used 73.2% fewer approximate
evidence words than full transcript. The development result agrees with the held-out
pattern.

### 4.3 Component checks

The standard-library test suite passed **23/23** tests. It covers explicit
fact handling, correction and conflict transitions, source preservation,
database reopen and clearing, serialization and ordering, unknown queries,
unsupported values, context budget, irrelevant history, adversarial evidence,
corruption, outage fallback, rule extraction, and fixed rendering.

## 5. Interpretation

The experiment demonstrates a simple point for this controlled task: preserving
conflict state prevents an unsafe latest-value answer, and explicit key/value
memory can use substantially less history text than full transcript processing
without losing task success. It does not show an accuracy gain over a full
transcript system; those systems tied at 100%. Cortex-M pays additional SQLite
and retrieval overhead and missed the strict relative latency target in this
run.

The 100% score depends on a fixed vocabulary, controlled sentence templates,
balanced categories, and a parser designed for those utterance forms. It must
not be generalized to arbitrary user conversation. The context measure is word
count rather than actual model tokens, and no model-token cost was measured
because the system intentionally uses no model.

## 6. Limitations and next experiments

- Expand the lexicon and rule grammar without changing the frozen v2 holdout.
- Add adversarial paraphrases, quoted speech, negation, topic shifts, typos,
  unknown values, and long-lived multi-session histories.
- Test on a new independently authored corpus after rule changes; do not tune
  against v2 holdout.
- Benchmark persistent storage growth and retrieval latency at larger scales.
- Keep M2M separate. Archived M2M-91/92/93 summaries do not meet the planned
  repeated-seed causal integration gate, and no fresh revalidation was run.

## 7. Reproducibility

From `E:\Research\Cortex-M` with Python 3.12.10:

```powershell
.\scripts\setup.ps1
.\scripts\python.ps1 -m unittest discover -s tests -v
.\scripts\python.ps1 data\generate.py
.\scripts\python.ps1 benchmarks\run_rule_benchmark.py --split dev
.\scripts\python.ps1 benchmarks\run_rule_benchmark.py --split holdout
```

The primary held-out case-level and summary artifacts are in
`results/rule-20261023T174812718078Z/`; the validated development artifacts
are in `results/rule-20261023T174811786987Z/`. Each includes its configuration,
runtime, dataset manifest, per-case output, aggregate summary, and
interpretation. The earlier pre-validation dev run is also retained separately
so the experiment history stays visible.
