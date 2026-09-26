"""Attention primitives implemented directly with PyTorch tensor operations."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class AttentionOutput:
    """Context vectors and the normalized weights used to create them."""

    values: Tensor
    weights: Tensor


def causal_attention_mask(
    sequence_length: int,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return a mask that lets each position attend only to itself and the past."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")
    return torch.ones(
        (sequence_length, sequence_length),
        dtype=torch.bool,
        device=device,
    ).tril()


def _validate_attention_inputs(query: Tensor, key: Tensor, value: Tensor) -> None:
    if query.ndim < 2 or key.ndim < 2 or value.ndim < 2:
        raise ValueError("query, key, and value must each have at least two dimensions")
    if query.shape[:-2] != key.shape[:-2] or key.shape[:-2] != value.shape[:-2]:
        raise ValueError("query, key, and value batch dimensions must match")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key feature dimensions must match")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError("key and value sequence lengths must match")
    if query.shape[-1] == 0:
        raise ValueError("query and key feature dimensions must be non-empty")
    if not query.is_floating_point() or not key.is_floating_point():
        raise TypeError("query and key must use floating-point dtypes")
    if query.dtype != key.dtype or key.dtype != value.dtype:
        raise TypeError("query, key, and value dtypes must match")
    if query.device != key.device or key.device != value.device:
        raise ValueError("query, key, and value must be on the same device")


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    attention_mask: Tensor | None = None,
) -> AttentionOutput:
    """Compute attention with matrix multiplication, scaling, and softmax."""
    _validate_attention_inputs(query, key, value)

    scores = query @ key.transpose(-2, -1)
    scores = scores / sqrt(query.shape[-1])
    if attention_mask is not None:
        if attention_mask.dtype != torch.bool:
            raise TypeError("attention_mask must use torch.bool dtype")
        if attention_mask.device != scores.device:
            raise ValueError("attention_mask must be on the same device as the inputs")
        try:
            attention_mask = torch.broadcast_to(attention_mask, scores.shape)
        except RuntimeError as error:
            raise ValueError(
                "attention_mask must broadcast to the attention score shape"
            ) from error
        if (~attention_mask).all(dim=-1).any():
            raise ValueError("attention_mask cannot hide every key for a query")
        scores = scores.masked_fill(~attention_mask, -torch.inf)
    weights = torch.softmax(scores, dim=-1)
    return AttentionOutput(values=weights @ value, weights=weights)


class MultiHeadSelfAttention(nn.Module):
    """Project, split, attend, and recombine multiple self-attention heads."""

    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be at least 1")
        if num_heads < 1:
            raise ValueError("num_heads must be at least 1")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

    def _split_heads(self, inputs: Tensor) -> Tensor:
        batch_size, sequence_length, _ = inputs.shape
        return inputs.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def forward(
        self,
        inputs: Tensor,
        *,
        padding_mask: Tensor | None = None,
        causal: bool = False,
    ) -> AttentionOutput:
        """Apply self-attention to ``batch x sequence x embedding`` inputs."""
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape batch x sequence x embedding")
        if inputs.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"input embedding dimension must be {self.embedding_dim}, "
                f"got {inputs.shape[-1]}"
            )
        if not inputs.is_floating_point():
            raise TypeError("inputs must use a floating-point dtype")

        batch_size, sequence_length, _ = inputs.shape
        attention_mask: Tensor | None = None
        if padding_mask is not None:
            if padding_mask.dtype != torch.bool:
                raise TypeError("padding_mask must use torch.bool dtype")
            if padding_mask.shape != (batch_size, sequence_length):
                raise ValueError("padding_mask must have shape batch x sequence")
            if padding_mask.device != inputs.device:
                raise ValueError("padding_mask must be on the same device as inputs")
            attention_mask = padding_mask[:, None, None, :]
        if causal:
            causal_mask = causal_attention_mask(
                sequence_length,
                device=inputs.device,
            )[None, None, :, :]
            attention_mask = (
                causal_mask
                if attention_mask is None
                else attention_mask & causal_mask
            )

        query = self._split_heads(self.query_projection(inputs))
        key = self._split_heads(self.key_projection(inputs))
        value = self._split_heads(self.value_projection(inputs))
        attended = scaled_dot_product_attention(
            query,
            key,
            value,
            attention_mask=attention_mask,
        )
        combined = attended.values.transpose(1, 2).contiguous().reshape(
            batch_size,
            sequence_length,
            self.embedding_dim,
        )
        output = self.output_projection(combined)
        if padding_mask is not None:
            output = output * padding_mask.unsqueeze(-1)
        return AttentionOutput(values=output, weights=attended.weights)
