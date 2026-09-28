# Changelog

All notable changes to this project are documented here. The project follows
Semantic Versioning for its Python and command-line interfaces.

## 1.0.0 — 2026-09-28

### Added

- Deterministic tensor, autograd, loss, optimizer, tokenizer, embedding, and
  causal-attention experiments with focused explanations and tests.
- A compact causal Transformer, CC0-licensed 24-document corpus, document-level
  split, shifted batching, and an isolated-seed AdamW training loop.
- Three reproducible training configurations with complete JSON loss curves and
  a dependency-free SVG rendering.
- An auditable greedy-generation demo that reports its prompt, generated tokens,
  full training configuration, loss endpoints, and limitations as JSON.
- Architecture, model card, AI-01 quality review, release checklist, and
  machine-readable release verification.

### Safety and limitations

- The model is trained on only six short documents and is not suitable for
  production use, factual assistance, decisions about people, or autonomous
  actions.
- The local loss and generation outputs demonstrate software mechanics; they do
  not establish language quality, robustness, fairness, privacy, or safety.
- The project has no checkpoint distribution, network service, content filter,
  factual grounding, monitoring, or production capacity claim.
