"""Measured OOV, padding, and sequence-length embedding experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch
from torch import Tensor

from tiny_transformer.embeddings import TransformerEmbedding
from tiny_transformer.tokenizer import BatchEncoding, Vocabulary


@dataclass(frozen=True)
class EmbeddingExperiment:
    """Artifacts from a fixed tokenizer and embedding pipeline."""

    vocabulary: Vocabulary
    batch: BatchEncoding
    embeddings: Tensor
    token_gradient_rows: int
    position_gradient_rows: int

    def summary(self) -> dict[str, object]:
        """Return JSON-compatible evidence for boundary behavior."""
        padding_vectors = self.embeddings[~self.batch.attention_mask]
        return {
            "vocabulary_size": len(self.vocabulary),
            "vocabulary": list(self.vocabulary.tokens),
            "token_ids": self.batch.token_ids.tolist(),
            "attention_mask": self.batch.attention_mask.tolist(),
            "original_lengths": list(self.batch.original_lengths),
            "encoded_shape": list(self.batch.token_ids.shape),
            "embedding_shape": list(self.embeddings.shape),
            "unknown_token_count": int(
                self.batch.token_ids.eq(self.vocabulary.unk_id).sum().item()
            ),
            "padding_token_count": int(
                self.batch.token_ids.eq(self.vocabulary.pad_id).sum().item()
            ),
            "truncated_sequence_count": sum(
                length > self.batch.token_ids.shape[1]
                for length in self.batch.original_lengths
            ),
            "padding_vector_l1": float(padding_vectors.abs().sum().item()),
            "token_gradient_rows": self.token_gradient_rows,
            "position_gradient_rows": self.position_gradient_rows,
        }


def run_embedding_experiment(seed: int = 11) -> EmbeddingExperiment:
    """Exercise OOV, padding, truncation, shape, and gradient behavior."""
    corpus = (
        "attention uses token embeddings",
        "embeddings encode token identity",
    )
    vocabulary = Vocabulary.build(corpus)
    batch = vocabulary.encode_batch(
        ["token mystery", "mystery attention token embeddings beyond", ""],
        max_length=4,
    )

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        layer = TransformerEmbedding(
            vocab_size=len(vocabulary),
            embedding_dim=6,
            max_sequence_length=4,
            padding_idx=vocabulary.pad_id,
        )
        embeddings = layer(batch.token_ids)
        embeddings.sum().backward()

        token_gradient = layer.tokens.weight.grad
        position_gradient = layer.positions.weight.grad
        if token_gradient is None or position_gradient is None:
            raise RuntimeError("embedding backward pass did not produce gradients")

        token_gradient_rows = int(token_gradient.abs().sum(dim=1).gt(0).sum().item())
        position_gradient_rows = int(
            position_gradient.abs().sum(dim=1).gt(0).sum().item()
        )

    return EmbeddingExperiment(
        vocabulary=vocabulary,
        batch=batch,
        embeddings=embeddings.detach(),
        token_gradient_rows=token_gradient_rows,
        position_gradient_rows=position_gradient_rows,
    )


def main() -> None:
    """Print the experiment as inspectable JSON."""
    print(json.dumps(run_embedding_experiment().summary(), indent=2))


if __name__ == "__main__":
    main()
