# Cortex M architecture v1

```text
conversation turns
      |
      +--> fact extractor --> evidence validation --> SQLite event store
                                                |
new request --> goal parse --> retrieval --------+
                              |
                              v
                     compact working gist
                              |
                      response plan/action
                              |
                   local language model
                              |
              evidence verifier / safe fallback
                              |
                         response
```

## State types

- `ConversationTurn`: turn index, role, text.
- `FactRecord`: stable id, normalized key, value, source turn, exact evidence
  text, explicitness, confidence, correction flag, and status (`active`,
  `conflicted`, or `superseded`).
- `EvidenceRef`: fact id, source turn, evidence text, and retrieval score.
- `WorkingGist`: current goal, active topics, selected evidence, unresolved
  questions, response action, and approximate token count capped at 512.
- `RetrievalResult`: ranked evidence, conflicts, and availability status.
- `ResponsePlan`: action (`answer`, `clarify`, `abstain`, `fallback`), goal,
  allowed evidence, uncertainty, and a short outline.

## State transition rules

Only explicit user statements enter persistent memory. A direct correction
supersedes earlier active values while retaining the old event as history.
Different explicit values without a correction mark both records conflicted;
the system asks which is current. Inferred facts can be shown as tentative
within the current turn but are not stored as user facts. Clearing a fact or
session removes its active records and records a clear event in the experiment
log.

## Retrieval, planning, and verification

Use normalized lexical BM25 ranking over user fact keys, values, and evidence
text, with exact key matches as a strong signal. Do not silently answer from a
low-scoring candidate. Build the gist from the current goal, top relevant
evidence, and conflict/uncertainty state. For memory-dependent questions, the
verifier accepts only responses supported by retrieved facts; on mismatch it
uses the planned clarification or abstention fallback. The verifier must not
use benchmark gold labels.

## Backend boundary

`TextBackend.generate(messages, config) -> GenerationResult` is the only model
interface. `GenerationResult` records text, prompt/completion token counts,
elapsed time, backend id, and stop reason. The local llama.cpp backend uses its
loopback OpenAI-compatible endpoint. Deterministic test backend and annotated
fact extractor isolate state-management logic from model quality.

M2M Sparsemax-K is not in the v1 response path. It remains an optional
relational-computation module behind a separate gate and benchmark.
