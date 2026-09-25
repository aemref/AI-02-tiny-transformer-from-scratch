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


class PositionalEmbedding(nn.Module):
    """Learn one vector for each position up to a fixed sequence limit."""

    def __init__(self, max_sequence_length: int, embedding_dim: int) -> None:
        super().__init__()
        if max_sequence_length < 1:
            raise ValueError("max_sequence_length must be at least 1")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be at least 1")
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(max_sequence_length, embedding_dim)

    @property
    def weight(self) -> Tensor:
        """Expose learned position vectors for inspection and tests."""
        return self.embedding.weight

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return position vectors matching a rank-two token-id batch."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape batch x sequence")
        batch_size, sequence_length = token_ids.shape
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds configured maximum "
                f"{self.max_sequence_length}"
            )
        positions = torch.arange(sequence_length, device=token_ids.device)
        return self.embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)


class TransformerEmbedding(nn.Module):
    """Combine learned token and position vectors and zero padded positions."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_sequence_length: int,
        *,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.padding_idx = padding_idx
        self.tokens = TokenEmbedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
        )
        self.positions = PositionalEmbedding(max_sequence_length, embedding_dim)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Add token and position representations under the padding mask."""
        combined = self.tokens(token_ids) + self.positions(token_ids)
        non_padding = token_ids.ne(self.padding_idx).unsqueeze(-1)
        return combined * non_padding
