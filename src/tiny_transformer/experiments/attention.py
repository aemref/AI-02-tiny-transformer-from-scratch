"""A deterministic causal-attention map experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch
from torch import Tensor

from tiny_transformer.attention import (
    causal_attention_mask,
    scaled_dot_product_attention,
)


@dataclass(frozen=True)
class AttentionExperiment:
    """Inputs and measured outputs from a small causal-attention calculation."""

    tokens: tuple[str, ...]
    representations: Tensor
    attention_map: Tensor
    context_vectors: Tensor

    def summary(self) -> dict[str, object]:
        """Return JSON-compatible evidence about normalization and causality."""
        probabilities = self.attention_map.clamp_min(torch.finfo(torch.float64).tiny)
        entropy = -(self.attention_map * probabilities.log()).sum(dim=-1)
        return {
            "tokens": list(self.tokens),
            "representation_shape": list(self.representations.shape),
            "attention_map": self.attention_map.tolist(),
            "row_sums": self.attention_map.sum(dim=-1).tolist(),
            "future_attention_total": float(
                torch.triu(self.attention_map, diagonal=1).sum().item()
            ),
            "most_attended_token_by_query": [
                self.tokens[index] for index in self.attention_map.argmax(dim=-1)
            ],
            "attention_entropy": entropy.tolist(),
            "context_vectors": self.context_vectors.tolist(),
        }


def run_attention_experiment() -> AttentionExperiment:
    """Generate a three-token attention map from fixed representations."""
    tokens = ("the", "cat", "sat")
    representations = torch.tensor(
        [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=torch.float64,
    )
    output = scaled_dot_product_attention(
        representations,
        representations,
        representations,
        attention_mask=causal_attention_mask(len(tokens)),
    )
    return AttentionExperiment(
        tokens=tokens,
        representations=representations,
        attention_map=output.weights,
        context_vectors=output.values,
    )


def main() -> None:
    """Print the measured attention map as inspectable JSON."""
    print(json.dumps(run_attention_experiment().summary(), indent=2))


if __name__ == "__main__":
    main()
