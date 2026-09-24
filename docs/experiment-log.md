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

