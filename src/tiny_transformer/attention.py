"""Attention primitives implemented directly with PyTorch tensor operations."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor


@dataclass(frozen=True)
class AttentionOutput:
    """Context vectors and the normalized weights used to create them."""

    values: Tensor
    weights: Tensor


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
