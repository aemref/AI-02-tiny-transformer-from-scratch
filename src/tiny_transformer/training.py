"""Deterministic full-batch training for the tiny language model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from tiny_transformer.corpus import NextTokenBatch
from tiny_transformer.model import TinyTransformerLanguageModel


@dataclass(frozen=True)
class TrainingConfig:
    """All values needed to reproduce one local training run."""

    steps: int = 40
    learning_rate: float = 0.02
    weight_decay: float = 0.0
    dropout: float = 0.0
    seed: int = 23
    embedding_dim: int = 16
    num_heads: int = 2
    num_layers: int = 1

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be at least 1")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")


@dataclass(frozen=True)
class TrainingRun:
    """Trained model and loss observations recorded after each update."""

    model: TinyTransformerLanguageModel
    train_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    config: TrainingConfig


def next_token_loss(logits: Tensor, target_ids: Tensor) -> Tensor:
    """Calculate cross entropy while ignoring padded target positions."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape batch x sequence x vocabulary")
    if target_ids.shape != logits.shape[:2]:
        raise ValueError("target_ids must match the first two logits dimensions")
    if target_ids.dtype != torch.long:
        raise TypeError("target_ids must use torch.long dtype")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1),
        ignore_index=-100,
    )


def _evaluate(
    model: TinyTransformerLanguageModel,
    batch: NextTokenBatch,
) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(batch.input_ids, padding_mask=batch.padding_mask)
        return float(next_token_loss(logits, batch.target_ids).item())


def train_language_model(
    train_batch: NextTokenBatch,
    validation_batch: NextTokenBatch,
    *,
    vocab_size: int,
    config: TrainingConfig | None = None,
) -> TrainingRun:
    """Train with AdamW while restoring the caller's random state afterward."""
    if config is None:
        config = TrainingConfig()
    if train_batch.input_ids.shape[1] != validation_batch.input_ids.shape[1]:
        raise ValueError("train and validation sequence lengths must match")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = TinyTransformerLanguageModel(
            vocab_size,
            embedding_dim=config.embedding_dim,
            max_sequence_length=train_batch.input_ids.shape[1],
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        train_losses = [_evaluate(model, train_batch)]
        validation_losses = [_evaluate(model, validation_batch)]

        for _ in range(config.steps):
            model.train()
            optimizer.zero_grad()
            logits = model(
                train_batch.input_ids,
                padding_mask=train_batch.padding_mask,
            )
            loss = next_token_loss(logits, train_batch.target_ids)
            loss.backward()
            optimizer.step()
            train_losses.append(_evaluate(model, train_batch))
            validation_losses.append(_evaluate(model, validation_batch))

    return TrainingRun(
        model=model,
        train_losses=tuple(train_losses),
        validation_losses=tuple(validation_losses),
        config=config,
    )
