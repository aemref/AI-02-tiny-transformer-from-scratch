# Experiment log

This log records commands that were actually run. It is not a forecast or a
benchmark of Transformer quality.

## 2026-09-24 — foundation concepts

Environment:

- macOS arm64
- Python 3.14.7
- PyTorch 2.14.0

Verified observations:

| Experiment | Command | Observation |
| --- | --- | --- |
| Tensor broadcasting | `python -m tiny_transformer.experiments.tensors` | A length-three scale broadcast over a `2 x 3` float32 matrix and preserved its shape. |
| Autograd | `python -m tiny_transformer.experiments.autograd` | At `weight = 0.5`, MSE was `10.5` and the computed weight gradient was `-14.0`. |
| Cross entropy | `python -m tiny_transformer.experiments.losses` | Two correctly classified rows produced a mean loss of `0.23989622723863246`. |
| SGD | `python -m tiny_transformer.experiments.optimizers` | In 80 steps, loss moved from `4.0` to `4.013882971776184e-06`; learned weight was `2.997154951095581` and bias was `-0.49999991059303284`. |

The observations above are deterministic learning examples on fixed, tiny
inputs. They do not measure model accuracy, throughput, or generalization.

## 2026-09-25 — tokenizer and embeddings

Environment:

- macOS arm64
- Python 3.14.7
- PyTorch 2.14.0

Command:

```bash
python -m tiny_transformer.experiments.embeddings
```

Verified observations from the fixed three-row batch:

- The vocabulary contained 8 entries, including reserved padding and unknown
  tokens.
- Encoding to a width of 4 produced a `3 x 4` id tensor and a `3 x 4 x 6`
  embedding tensor.
- The encoded batch contained 2 unknown-token ids and 6 padding ids.
- One input was truncated because its original length was 5.
- The absolute sum of all emitted padding vectors was `0.0`.
- Backpropagation reached 4 token-table rows and all 4 used position rows.

These observations cover tokenizer and embedding boundaries only. They do not
measure language-model accuracy or claim that the chosen vocabulary is useful
for a larger corpus.

## 2026-09-26 — self-attention

Environment:

- macOS arm64
- Python 3.14.7
- PyTorch 2.14.0

Command:

```bash
python -m tiny_transformer.experiments.attention
```

Verified observations from three fixed two-dimensional representations:

- The causal attention map had shape `3 x 3`, and every row summed to `1.0`.
- The sum of all future-position weights above the diagonal was `0.0`.
- Query rows placed their largest weight on `the`, `cat`, and `cat`,
  respectively; the final row had an equal `0.4011120926797859` weight for
  `cat` and `sat`, so `argmax` selected the earlier index.
- The three context vectors were `[1.0, 0.0]`,
  `[1.0, 0.6697615493266569]`, and
  `[0.5988879073202141, 0.8022241853595719]`.

This is deterministic contract evidence for normalization and masking, not a
benchmark or an explanation of learned model behavior. See the
[attention note](concepts/attention.md) for interpretation limits.
