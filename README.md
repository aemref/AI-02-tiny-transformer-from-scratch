# AI-02 Tiny Transformer From Scratch

A test-first exploration of the tensor, gradient, optimization, tokenization,
attention, and training mechanics behind a small Transformer in PyTorch.

The repository begins with isolated experiments. Each concept must have an
executable example, a focused test, and a visual explanation before it becomes
part of the model.

## Concept experiments

| Concept | Executable module | Visual note |
| --- | --- | --- |
| Tensor shape, dtype, broadcasting | `tiny_transformer.experiments.tensors` | [Tensors](docs/concepts/tensors.md) |
| Reverse-mode differentiation | `tiny_transformer.experiments.autograd` | [Autograd](docs/concepts/autograd.md) |
| Cross entropy from logits | `tiny_transformer.experiments.losses` | [Losses](docs/concepts/losses.md) |
| SGD training step | `tiny_transformer.experiments.optimizers` | [Optimizers](docs/concepts/optimizers.md) |
| Tokenizer and embeddings | `tiny_transformer.experiments.embeddings` | [Tokenizer and embeddings](docs/concepts/embeddings.md) |
| Causal self-attention | `tiny_transformer.experiments.attention` | [Self-attention](docs/concepts/attention.md) |

Each module prints JSON so its measured values can be inspected or captured by
another tool. The first verified run is recorded in the
[experiment log](docs/experiment-log.md); the remaining monthly build stages
are tracked in the [technical roadmap](docs/roadmap.md).

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

## Scope

The project is educational and uses small local examples. It does not download
datasets, call hosted models, or claim benchmark results that were not run.

## License

MIT
