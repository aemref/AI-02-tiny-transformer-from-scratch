# Technical roadmap

This repository follows the AI-02 technical scope in the portfolio roadmap.
Human-participation goals and unrelated certification work are intentionally
outside this plan.

## 1. Foundation concepts

- [x] Create an installable PyTorch project with tests and linting.
- [x] Add deterministic tensor, autograd, loss, and optimizer experiments.
- [x] Give every concept a focused test and visual explanation.
- [x] Establish at least ten passing tests and an experiment log.

## 2. Tokenizer and embeddings

- [x] Build a small tokenizer and explicit vocabulary pipeline.
- [x] Implement token and positional embeddings.
- [x] Test shape, gradient, padding, unknown-token, and sequence-length cases.
- [x] Record measured OOV, padding, and sequence-length experiments.

## 3. Self-attention

- [ ] Implement scaled dot-product attention from tensor operations.
- [ ] Add causal masking and multi-head attention.
- [ ] Test shapes, masks, invalid inputs, and gradient flow.
- [ ] Produce an attention map with interpretation limits.

## 4. Training loop

- [ ] Train a tiny language model on a small, licensed local text dataset.
- [ ] Provide a one-command deterministic training path.
- [ ] Run and record overfitting, regularization, and learning-rate experiments.
- [ ] Publish the configuration, loss curves, and reproduction commands.

## 5. Release

- [ ] Write the model card and technical architecture document.
- [ ] Add an auditable local demo and cross-project quality review.
- [ ] Run every release gate from a clean clone.
- [ ] Publish an immutable `v1.0.0` tag and GitHub Release.
