"""A deterministic optimization loop for a one-layer regression model."""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class OptimizationRun:
    """Measured state from a fixed number of gradient-descent steps."""

    losses: tuple[float, ...]
    learned_weight: float
    learned_bias: float
    final_gradient: Tensor

    def summary(self) -> dict[str, object]:
        """Return compact, JSON-compatible optimization evidence."""
        return {
            "steps": len(self.losses),
            "initial_loss": self.losses[0],
            "final_loss": self.losses[-1],
            "learned_weight": self.learned_weight,
            "learned_bias": self.learned_bias,
            "final_weight_gradient": self.final_gradient.item(),
        }


def fit_linear_model(steps: int = 80, learning_rate: float = 0.1) -> OptimizationRun:
    """Fit ``target = 3 * input - 0.5`` using SGD and mean squared error."""
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    inputs = torch.linspace(-1.0, 1.0, steps=9).unsqueeze(-1)
    targets = 3.0 * inputs - 0.5
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    losses: list[float] = []

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(inputs), targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if model.weight.grad is None:  # pragma: no cover - defensive invariant
        raise RuntimeError("training ended without a weight gradient")

    return OptimizationRun(
        losses=tuple(losses),
        learned_weight=model.weight.detach().item(),
        learned_bias=model.bias.detach().item(),
        final_gradient=model.weight.grad.detach().clone(),
    )


def main() -> None:
    """Print the experiment as inspectable JSON."""
    print(json.dumps(fit_linear_model().summary(), indent=2))


if __name__ == "__main__":
    main()

