"""Token and position representation layers for the tiny Transformer."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TokenEmbedding(nn.Module):
    """Look up learned token vectors while keeping padding fixed at zero."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        *,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        if vocab_size < 2:
            raise ValueError("vocab_size must include at least two tokens")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be at least 1")
        if not 0 <= padding_idx < vocab_size:
            raise ValueError("padding_idx must be inside the vocabulary")

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

    @property
    def weight(self) -> Tensor:
        """Expose the learned table for inspection in focused experiments."""
        return self.embedding.weight

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return ``batch x sequence x embedding`` token representations."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape batch x sequence")
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must use torch.long dtype")
        return self.embedding(token_ids)
