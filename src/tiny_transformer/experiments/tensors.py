"""Deterministic tensor creation, shape, dtype, and broadcasting experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class TensorExperiment:
    """Values in a row-wise scaling experiment."""

    inputs: Tensor
    scale: Tensor
    scaled: Tensor

    def summary(self) -> dict[str, object]:
        """Return JSON-compatible evidence about the tensor operation."""
        return {
            "input_shape": list(self.inputs.shape),
            "scale_shape": list(self.scale.shape),
            "output_shape": list(self.scaled.shape),
            "dtype": str(self.scaled.dtype),
            "scaled_values": self.scaled.tolist(),
        }


def build_tensor_experiment(seed: int = 7) -> TensorExperiment:
    """Create a repeatable matrix and scale its columns by broadcasting."""
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn((2, 3), generator=generator, dtype=torch.float32)
    scale = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
    return TensorExperiment(inputs=inputs, scale=scale, scaled=inputs * scale)


def main() -> None:
    """Print the experiment as inspectable JSON."""
    print(json.dumps(build_tensor_experiment().summary(), indent=2))


if __name__ == "__main__":
    main()

