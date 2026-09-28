# AI-02 Tiny Transformer From Scratch

[![CI](https://github.com/aemref/AI-02-tiny-transformer-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/aemref/AI-02-tiny-transformer-from-scratch/actions/workflows/ci.yml)

A test-first, line-by-line exploration of the tensor, gradient, tokenization,
causal-attention, training, and decoding mechanics behind a tiny Transformer in
PyTorch. The project is local, deterministic, and deliberately too small for
real language use.

## What is included

- A whitespace tokenizer, explicit vocabulary, token/position embeddings, and
  causal multi-head self-attention implemented from tensor operations.
- A compact decoder-only model with tied token/output weights and a deterministic
  full-batch AdamW training loop.
- A self-authored, 24-document CC0 corpus with document-level splitting and
  next-token batching that never crosses document boundaries.
- Three checked-in 60-step experiments with full-precision JSON curves and a
  dependency-free SVG visualization.
- An auditable local generation demo that retrains its disclosed configuration
  and emits machine-readable JSON.

## Concept experiments

| Concept | Executable module | Visual note |
| --- | --- | --- |
| Tensor shape, dtype, broadcasting | `tiny_transformer.experiments.tensors` | [Tensors](docs/concepts/tensors.md) |
| Reverse-mode differentiation | `tiny_transformer.experiments.autograd` | [Autograd](docs/concepts/autograd.md) |
| Cross entropy from logits | `tiny_transformer.experiments.losses` | [Losses](docs/concepts/losses.md) |
| SGD training step | `tiny_transformer.experiments.optimizers` | [Optimizers](docs/concepts/optimizers.md) |
| Tokenizer and embeddings | `tiny_transformer.experiments.embeddings` | [Tokenizer and embeddings](docs/concepts/embeddings.md) |
| Causal self-attention | `tiny_transformer.experiments.attention` | [Self-attention](docs/concepts/attention.md) |
| Language-model training | `tiny_transformer.experiments.training` | [Training experiments](docs/concepts/training.md) |

Each module prints JSON so measured values can be inspected or captured by
another tool. Verified runs are recorded in the
[experiment log](docs/experiment-log.md). See the
[architecture](docs/architecture.md), [model card](docs/model-card.md), and
[technical roadmap](docs/roadmap.md) for boundaries and design decisions.

## Development

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
ruff check .
```

Run one experiment directly, for example:

```bash
python -m tiny_transformer.experiments.optimizers
```

Run all three deterministic training configurations and regenerate their JSON
evidence and SVG loss curves with one command:

```bash
python -m tiny_transformer.experiments.training --output-dir docs/experiments
```

## Local demo

Train the fixed lower-rate configuration and greedily emit six tokens:

```bash
python -m tiny_transformer.experiments.demo \
  --prompt "attention uses" \
  --tokens 6
```

The output includes tokenized input, generated tokens, context size, seed,
hyperparameters, corpus counts, and train/validation loss endpoints. There is no
downloaded checkpoint, hidden service, or hand-edited output.

## Measured result

The high-rate unregularized run reduced training loss from `12.0777` to `0.0320`,
while its best validation loss was `8.9956` at step 12 and worsened to `12.2000`
by step 60. The regularized run did not improve best validation loss. The
lower-rate validation curve was still improving at step 60. These observations
demonstrate the mechanics of overfitting; they do not establish language quality.

The full configurations and curves are in
[`training-results.json`](docs/experiments/training-results.json) and
[`training-loss-curves.svg`](docs/experiments/training-loss-curves.svg).

## Release verification

The machine-readable repository gate checks application version, corpus license,
experiment artifacts, and required release documents:

```bash
python -m tiny_transformer.release
```

The full clean-clone sequence is versioned in the
[release checklist](docs/release-checklist.md), and observed results are recorded
in the release evidence. CI runs the public experiments, demo, release check,
tests, lint, bytecode compilation, and wheel build.

## Limitations

The bundled 24-document synthetic corpus is released under CC0; its provenance
is recorded beside the data. Six short training documents, a whitespace
tokenizer, and a six-token context cannot support useful language generation.
Unknown words collapse to one token, greedy decoding can repeat, and there is no
safety filter, factual grounding, privacy evaluation, or fairness evaluation.
This project is educational and is not suitable for production use.

## License

MIT
