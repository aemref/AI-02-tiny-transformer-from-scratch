# v1.0.0 Release Checklist

The `1.0.0` version identifies the public Python, CLI, experiment, and evidence
contract. The release tag is created only after every gate passes from a clean
`main` clone and the remote branch is rechecked for divergence.

## Source and evidence gates

- [x] The 24-document corpus has CC0 license and provenance metadata.
- [x] Document splitting, next-token batching, causal masking, gradients,
  training determinism, and sliding-window generation have focused tests.
- [x] Three controlled 60-step runs publish complete JSON curves and an SVG.
- [x] README, architecture, model card, experiment log, cross-project review,
  limitations, and reproduction commands are present.
- [x] The local demo retrains a disclosed configuration and emits auditable JSON.
- [x] Release readiness fails closed on version, corpus, evidence, and documents.
- [x] Clean-clone verification passes with the commands below.
- [x] Local `main`, `origin/main`, the annotated tag, and GitHub Release agree.

## Clean-clone verification

Run these gates in order from a fresh clone of the release commit:

```bash
ruff check .
python -m pytest
python -m tiny_transformer.experiments.tensors
python -m tiny_transformer.experiments.autograd
python -m tiny_transformer.experiments.losses
python -m tiny_transformer.experiments.optimizers
python -m tiny_transformer.experiments.embeddings
python -m tiny_transformer.experiments.attention
python -m tiny_transformer.experiments.training --output-dir /tmp/ai02-training
python -m tiny_transformer.experiments.demo --prompt "attention uses" --tokens 6
python -m tiny_transformer.release
python -m compileall -q src tests
python -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
git diff --check
```

Compare the regenerated training JSON and SVG byte-for-byte with the tracked
artifacts. Inspect the wheel contents and confirm that Python modules, corpus,
and corpus license are included. Finish with a clean worktree.

## Release policy

Do not tag or push if a gate fails, generated evidence changes unexpectedly,
history diverges, or the worktree contains unexplained files. Never rewrite the
tag, force-push, weaken a gate to make it pass, or claim model quality from this
small synthetic experiment.
