# AI-02 Tiny Transformer From Scratch

A test-first exploration of the tensor, gradient, optimization, tokenization,
attention, and training mechanics behind a small Transformer in PyTorch.

The repository begins with isolated experiments. Each concept must have an
executable example, a focused test, and a visual explanation before it becomes
part of the model.

## Development

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
ruff check .
```

## Scope

The project is educational and uses small local examples. It does not download
datasets, call hosted models, or claim benchmark results that were not run.

## License

MIT

