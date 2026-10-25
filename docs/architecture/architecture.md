# Cortex-M architecture — model-free version

```text
user text
   |
   v
configured exact-phrase rules -----> unknown / unsupported -> abstain
   |
   v
explicit facts + source text
   |
   v
SQLite event store -----> lexical key retrieval
   |                              |
   +-- corrections/conflicts -----+
                                  v
                       answer / clarify / abstain / fallback
                                  |
                                  v
                       fixed evidence-bound template
```

## Input boundary

`RuleFactExtractor` accepts a configured list of fact keys and values. It
recognizes exact lexical matches and correction cues, plus a built-in explicit
`my name is <name>` field whose value does not need to be enumerated in the
lexicon. It does not infer a missing value, paraphrase unknown terms, or use an
AI model. Other vocabulary coverage requires explicit lexicon and rule work.
Unrecognized input produces no fact.

## Stored state

- `FactRecord`: stable id, key, value, source turn, exact source text,
  explicitness, confidence, correction flag, and state.
- A direct correction supersedes older active records.
- Conflicting explicit values retain both sides and trigger clarification.
- Inferred items are rejected from persistent memory.
- Clearing records an audit event.

## Retrieval and response

BM25 ranks the stored facts. An exact key match limits results to that key so
unrelated conflicts do not affect the answer. `plan_response` selects answer,
clarify, abstain, or fallback. `render_response` turns that plan into a fixed
sentence using only planned evidence.

## Execution boundary

The main path uses Python standard library only: SQLite, regular expressions,
hashing, and fixed string templates. The codebase contains no model client and
makes no network calls.

## Known limits

This design is suitable for known fields and controlled phrasing. It is not a
general conversation model. Unknown vocabulary, paraphrases, sarcasm, quoted
speech, and complex multi-fact sentences can be missed or misread. The safe
behavior is to ask the user to restate or confirm the detail.
