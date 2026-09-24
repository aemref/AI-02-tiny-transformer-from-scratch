"""A minimal reverse-mode automatic differentiation experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class AutogradExperiment:
    """Forward values and the gradient produced by one backward pass."""

    inputs: Tensor
    targets: Tensor
    weight: Tensor
    predictions: Tensor
    loss: Tensor
    weight_gradient: Tensor

    def summary(self) -> dict[str, object]:
        """Return compact evidence for the forward and backward passes."""
        return {
            "weight": self.weight.item(),
            "predictions": self.predictions.tolist(),
            "loss": self.loss.item(),
            "weight_gradient": self.weight_gradient.item(),
        }


def compute_weight_gradient(weight_value: float = 0.5) -> AutogradExperiment:
    """Differentiate mean squared error for the model ``prediction = w * x``."""
    inputs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    targets = 2.0 * inputs
    weight = torch.tensor(weight_value, dtype=torch.float64, requires_grad=True)

    predictions = weight * inputs
    loss = torch.mean((predictions - targets) ** 2)
    loss.backward()

    if weight.grad is None:  # pragma: no cover - defensive invariant
        raise RuntimeError("autograd did not populate the weight gradient")

    return AutogradExperiment(
        inputs=inputs,
        targets=targets,
        weight=weight.detach(),
        predictions=predictions.detach(),
        loss=loss.detach(),
        weight_gradient=weight.grad.detach().clone(),
    )


def main() -> None:
    """Print the experiment as inspectable JSON."""
    print(json.dumps(compute_weight_gradient().summary(), indent=2))


if __name__ == "__main__":
    main()

