"""A compact causal Transformer language model built from local primitives."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tiny_transformer.attention import MultiHeadSelfAttention
from tiny_transformer.embeddings import TransformerEmbedding


class TransformerBlock(nn.Module):
    """Pre-normalized causal attention and feed-forward residual block."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        *,
        feed_forward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadSelfAttention(embedding_dim, num_heads)
        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, feed_forward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: Tensor, *, padding_mask: Tensor) -> Tensor:
        """Transform inputs while keeping padded representations at zero."""
        attention = self.attention(
            self.attention_norm(inputs),
            padding_mask=padding_mask,
            causal=True,
        ).values
        hidden = inputs + self.attention_dropout(attention)
        hidden = hidden + self.feed_forward(self.feed_forward_norm(hidden))
        return hidden * padding_mask.unsqueeze(-1)


class TinyTransformerLanguageModel(nn.Module):
    """Predict the next token with a stack of causal Transformer blocks."""

    def __init__(
        self,
        vocab_size: int,
        *,
        embedding_dim: int,
        max_sequence_length: int,
        num_heads: int,
        num_layers: int = 1,
        feed_forward_dim: int | None = None,
        dropout: float = 0.0,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")
        if feed_forward_dim is None:
            feed_forward_dim = embedding_dim * 2
        if feed_forward_dim < 1:
            raise ValueError("feed_forward_dim must be at least 1")

        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.embeddings = TransformerEmbedding(
            vocab_size,
            embedding_dim,
            max_sequence_length,
            padding_idx=padding_idx,
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(
                embedding_dim,
                num_heads,
                feed_forward_dim=feed_forward_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.output.weight = self.embeddings.tokens.embedding.weight

    def forward(
        self,
        token_ids: Tensor,
        *,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Return vocabulary logits shaped batch x sequence x vocabulary."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape batch x sequence")
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must use torch.long dtype")
        if padding_mask is None:
            padding_mask = token_ids.ne(self.padding_idx)
        if padding_mask.dtype != torch.bool:
            raise TypeError("padding_mask must use torch.bool dtype")
        if padding_mask.shape != token_ids.shape:
            raise ValueError("padding_mask must match token_ids shape")
        if padding_mask.device != token_ids.device:
            raise ValueError("padding_mask must be on the same device as token_ids")
        if (~padding_mask).all(dim=-1).any():
            raise ValueError("each sequence must contain at least one real token")

        hidden = self.embeddings(token_ids)
        for block in self.blocks:
            hidden = block(hidden, padding_mask=padding_mask)
        hidden = self.output_norm(hidden) * padding_mask.unsqueeze(-1)
        return self.output(hidden)
