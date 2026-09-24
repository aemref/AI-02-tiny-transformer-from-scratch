"""Cross-entropy loss experiment for a tiny token-classification batch."""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class CrossEntropyExperiment:
    """Logits, labels, and loss values for a classification batch."""

    logits: Tensor
    targets: Tensor
    per_example_loss: Tensor
    mean_loss: Tensor

    def summary(self) -> dict[str, object]:
        """Return JSON-compatible loss evidence."""
        return {
            "logits_shape": list(self.logits.shape),
            "targets": self.targets.tolist(),
            "predictions": self.logits.argmax(dim=-1).tolist(),
            "per_example_loss": self.per_example_loss.tolist(),
            "mean_loss": self.mean_loss.item(),
        }


def evaluate_cross_entropy(
    logits: Tensor | None = None,
    targets: Tensor | None = None,
) -> CrossEntropyExperiment:
    """Evaluate raw class logits with numerically stable cross entropy."""
    if logits is None:
        logits = torch.tensor(
            [[2.5, 0.2, -1.0], [0.1, 1.8, 0.4]],
            dtype=torch.float64,
        )
    if targets is None:
        targets = torch.tensor([0, 1], dtype=torch.long)

    per_example_loss = F.cross_entropy(logits, targets, reduction="none")
    return CrossEntropyExperiment(
        logits=logits.detach().clone(),
        targets=targets.detach().clone(),
        per_example_loss=per_example_loss.detach(),
        mean_loss=per_example_loss.mean().detach(),
    )


def main() -> None:
    """Print the experiment as inspectable JSON."""
    print(json.dumps(evaluate_cross_entropy().summary(), indent=2))


if __name__ == "__main__":
    main()

