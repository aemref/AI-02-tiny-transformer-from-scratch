# Technical Architecture

## System boundary

AI-02 is a local educational language-model package. It loads a bundled
CC0-licensed corpus, creates a deterministic document split, trains a compact
causal Transformer, records loss curves, and can greedily generate a short token
sequence. It has no network call, hosted model, secret, telemetry service, or
paid infrastructure dependency.

```mermaid
flowchart LR
    C[24 CC0 text documents] --> S[Seeded document split]
    S --> V[Deterministic vocabulary]
    V --> B[Shifted next-token batches]
    B --> E[Token + position embeddings]
    E --> A[Causal multi-head self-attention]
    A --> F[Feed-forward residual block]
    F --> P[Tied vocabulary projection]
    P --> L[Masked cross-entropy]
    L --> O[AdamW update]
    P --> G[Greedy local demo]
    O --> J[JSON loss evidence]
    J --> R[Dependency-free SVG curves]
```

## Data and training path

1. `load_tiny_corpus` reads non-empty lines from the packaged corpus.
2. `split_corpus` shuffles document indexes with a local seeded generator so
   validation text stays separate without consuming the caller's random state.
3. `Vocabulary.build` orders tokens by descending frequency and then
   lexicographically. IDs `0` and `1` are reserved for padding and unknown text.
4. `build_next_token_batch` creates shifted input/target windows inside each
   document. Windows never cross document boundaries; target padding uses the
   cross-entropy ignore value `-100`.
5. `TinyTransformerLanguageModel` applies learned token and position embeddings,
   pre-normalized causal attention, and a feed-forward residual block. Its output
   projection shares the token-embedding weights.
6. `train_language_model` uses full-batch AdamW under an isolated PyTorch random
   state and records train and validation loss after every update.

## Inference path

The demo retrains one disclosed configuration from the bundled corpus on every
invocation; there is no hidden checkpoint. `greedy_generate` tokenizes the
prompt, retains only the model's most recent context window, chooses the highest
logit, and excludes the two reserved tokens. The JSON response contains the
prompt tokens, generated tokens, context size, complete training configuration,
loss endpoints, corpus counts, and a limitation statement.

Greedy decoding is deterministic for fixed source, dependencies, seed, and
hardware behavior. Exact floating-point values are not promised across all
PyTorch versions or devices.

## Trust and failure boundaries

| Boundary | Enforced behavior | Remaining limitation |
| --- | --- | --- |
| Packaged corpus | Reject fewer than two non-empty documents | License metadata does not establish linguistic quality |
| Corpus split | Keep whole documents in one partition | Vocabulary construction sees all corpus tokens |
| Batching | Reject empty inputs and prevent cross-document windows | Small overlapping windows are highly correlated |
| Attention | Enforce boolean masks, shapes, causality, and finite gradients | Unit tests do not prove learned interpretation |
| Training | Validate hyperparameters, isolate RNG state, record every loss | One fixed split cannot establish generalization |
| Generation | Reject empty prompts and reserved-token output | Unknown prompt words collapse to one token; no safety filter |
| Evidence | Reproduce JSON and SVG from source | Local loss curves are not a language-quality benchmark |

## Packaging and rollback

The wheel contains Python modules plus the corpus and its license note. CI runs
lint, tests, all concept experiments, the local demo, and training-artifact
reproduction. A release is eligible only after those gates also pass from a clean
clone. Rollback means checking out an earlier immutable tag; the project has no
mutable model registry or deployed service to roll back.

## Deliberate exclusions

The repository does not claim useful language generation, production inference,
fairness, safety, privacy, factuality, multilingual coverage, or capacity. It
does not provide sampling controls, checkpoint persistence, a web API, GPU
optimization, distributed training, or external evaluation. Those omissions are
intentional so the implementation remains small enough to audit line by line.
