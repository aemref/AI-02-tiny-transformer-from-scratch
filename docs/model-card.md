# Model Card: Tiny Causal Transformer

Status: educational mechanics model; not suitable for production use.

## Model details

AI-02 is a decoder-only next-token model assembled from repository-local
components. The measured configurations use one pre-normalized Transformer
block, two attention heads, a 16-dimensional embedding, a 32-dimensional
feed-forward path, learned positions, tied input/output token weights, and a
six-token context window. The tokenizer lowercases and splits on whitespace.

No pretrained weights are used. The demo trains from its disclosed seed on each
invocation and performs greedy decoding. The repository does not publish a model
checkpoint because the learning objective is to reproduce the implementation
and experiment, not distribute a capable text model.

## Intended use

The model is intended for learning, source inspection, tests, and demonstrations
of embeddings, causal attention, next-token loss, optimization, overfitting, and
deterministic decoding. It must not be used for factual assistance, decisions
about people, moderation, autonomous actions, or any production workload.

## Training data

The package contains 24 short synthetic English documents written for this
project and dedicated to the public domain under CC0 1.0. The experiments use a
seeded 18-document training / 6-document validation split, then deliberately
restrict fitting to the first six training documents to make overfitting visible.
The vocabulary has 132 entries, including padding and unknown tokens.

Vocabulary construction observes tokens from all 24 documents. Validation token
order and targets remain unseen during training, but the vocabulary exposure
means these results must not be treated as a clean estimate of generalization.

## Evaluation

Three 60-update, full-batch AdamW runs were measured on the same split. Loss is
token-level cross-entropy; lower is better. No accuracy or human-quality score is
claimed.

| Configuration | Initial train | Final train | Best validation (step) | Final validation |
| --- | ---: | ---: | ---: | ---: |
| Learning rate `0.03` | 12.0777 | 0.0320 | 8.9956 (12) | 12.2000 |
| LR `0.03`, dropout `0.2`, weight decay `0.02` | 12.0777 | 0.0437 | 9.0013 (13) | 12.0564 |
| Learning rate `0.003` | 12.0777 | 1.6941 | 9.6391 (60) | 9.6391 |

The high-rate configurations visibly overfit: training loss approaches zero
while validation loss rises. The tested regularization does not improve best
validation loss. The lower-rate curve is still improving at step 60, so the
experiment does not identify an optimal configuration.

## Limitations and risks

- Six short training documents cannot support coherent or reliable language.
- Whitespace tokenization loses punctuation structure and maps every unseen word
  to the same unknown token.
- A six-token context cannot represent long dependencies.
- Greedy decoding suppresses reserved tokens but has no repetition control,
  content filter, stop token, factual grounding, or uncertainty estimate.
- One fixed split and seed do not measure robustness, variance, fairness,
  multilingual behavior, memorization, or privacy.
- Loss values are sensitive to software, floating-point, and hardware details.
- Generated text may be repetitive, nonsensical, or misleading and must not be
  interpreted as knowledge or advice.

## Reproducibility

Run the full experiment and local demo from an installed checkout:

```bash
python -m tiny_transformer.experiments.training --output-dir docs/experiments
python -m tiny_transformer.experiments.demo --prompt "attention uses" --tokens 6
```

The checked-in full-precision curves live in
`docs/experiments/training-results.json`. The architecture, data contract, seed,
optimizer, configurations, split sizes, and reproduction command are versioned
with the code. Reproduction demonstrates a deterministic software path; it does
not validate the model for any real-world use.
