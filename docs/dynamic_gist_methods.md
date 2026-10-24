# Dynamic gist methods and routing policy

**Purpose:** Decide how Cortex-M should construct task context for each request without choosing a fixed gist label or fixed prompt summary. This is an implementation roadmap, not a claim that all methods are already built.

## Gist sources and methods

| Method | What it carries | Strength | Failure / current status | Best use |
|---|---|---|---|---|
| Current-turn state | Current request, explicit constraints, desired output | Always current; zero persistence needed | Existing core mostly routes by a small question/statement rule; does not understand arbitrary intents | Every request as the top-priority source |
| Provenance-backed fact gist | Structured fact, source sentence, turn, timestamp, confidence, correction/conflict state | Auditable and supports exact recall | Current rule extractor covers narrow English patterns; no broad automatic extraction | Stable personal/project facts |
| Query-focused episodic retrieval | Relevant prior turns/examples ranked for this goal | Grows with history and query; can return multiple examples | Current BM25 sees stored facts only and scans all rows; lexical mismatch misses paraphrases | Recall, ongoing project state |
| Extractive gist | Original sentences chosen by term salience, diversity/MMR, recency and budget | No model required; keeps source wording and provenance | Does not paraphrase or infer; lexical salience can select misleading lines | Shortening long transcripts safely |
| Hierarchical rolling gist | Recent-turn buffer plus session/day/project summaries linked to source events | Bounded context across long sessions; preserves recent details | Needs update, invalidation, and correction propagation; not yet built | Long-running projects/conversations |
| Support-set gist | A variable-size set of demonstrated input/output examples and task constraints | Directly supplies each task's evidence without naming a fixed mode | A finite candidate-rule baseline is implemented; the trainable neural set encoder and held-out episode evaluation remain open | User teaches a new numeric transformation by examples |
| Artifact/tool-state gist | File paths and hashes, open questions, plan, test results, errors, tool outputs | Grounds coding/research actions in real workspace state | No general tool agent or code sandbox currently exists | Coding, data and research tasks |
| Semantic learned gist | Embeddings or a language model's structured summary | Handles paraphrases and semantic relevance | Requires model compute, privacy/resource testing, calibration, and independent eval; no model selected | Broad natural-language tasks if the resource gate approves |
| User-validated gist | Ask user to confirm or correct uncertain compressed state | Resolves ambiguity and reduces silent drift | Adds interaction cost | Low-confidence, conflicting, or high-impact memory |

## Dynamic per-request policy

For every new prompt, construct candidate context from the current turn, matching memory, recent episode state, relevant examples, and available tool/artifact state. Select the representation by task needs and available evidence; include source IDs and freshness. Rank with query match, explicitness, recency, confidence, contradiction and token/compute budget. Keep raw evidence for every compressed claim. Preserve conflicting candidates rather than selecting one silently. If task type is unknown, use the current prompt and a small evidence-backed retrieval set, then ask or abstain where the system cannot ground a solver input.

The selector itself must be evaluated. A prompt parser or rule table that always maps requests into a fixed gist slot would only move the hardcoding to another layer.

## Chosen sequence

1. **Signal-injection positive control (complete with a robustness failure):** train MIMO with a task-mode vector and compare true, absent, and flipped vectors on identical held-out inputs. The selected context-pooling model reached 95.83% mean exact graph with true gist (9/10 seeds at 100%, one at 58.33%); no-gist was 0.017%, and flipped gist was 0%. A direct gist-to-route score variant averaged only 74.95% and was retained as a failed model variant. Results are in `results/mimo-gist-conditioned/` and `results/mimo-gist-direct-route/`.
2. **Memory-to-solver path (narrow schema implemented):** for each `mimo_gist_n8` request, Cortex-M retrieves the current `mode` fact and passes its encoded value to the trained router. Correcting mode alpha to beta changes the vector, selected routes and numeric output in paired tests. Missing or conflicted mode causes abstention before inference. The current runtime schema is still limited to alpha/beta; this is not a general-language gist parser.
3. **Variable support-set gist (baseline implemented; learned encoder next):** `CortexMIMOSystem` accepts a variable number of support input/output examples under `task=support_gist_route`. The transparent non-neural router scores 168 declared sum/product/difference/affine relations and uses Sparsemax to select per-output routes. This avoids alpha/beta mode labels, but still has a finite operation library and is not the trained MIMO network. Next encode examples as a permutation-invariant set in a trainable MIMO and evaluate held-out functions/route graphs/support sizes. Compare no examples, true examples, shuffled examples, and the non-neural baseline.
4. **Mixed gist policy:** combine current-turn constraints, retrieved facts, support examples, and optional tool state; choose/weight them dynamically and retain provenance.
5. **Broad language evaluation:** only after a language solver is selected and resource/privacy constraints are explicit; compare extractive, structured, and semantic gists on independently authored conversations.

## Required evidence

For each stage report paired task success, route/topology accuracy when known, output error, gist retrieval precision/recall, performance with missing/conflicted/stale gist, inference/training cost, and per-seed/per-episode failures. A causal gist claim requires that perturbing only the retrieved gist changes the routes and output in the predicted direction, while restoring the correct gist restores performance. Synthetic fixed families do not establish universal task capability.
