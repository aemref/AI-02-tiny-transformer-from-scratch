"""Licensed local corpus loading and next-token batch construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib.resources import files

import torch
from torch import Tensor

from tiny_transformer.tokenizer import Vocabulary


@dataclass(frozen=True)
class CorpusSplit:
    """Document-level train and validation partitions."""

    train: tuple[str, ...]
    validation: tuple[str, ...]


@dataclass(frozen=True)
class NextTokenBatch:
    """Padded inputs, shifted targets, and real-token positions."""

    input_ids: Tensor
    target_ids: Tensor
    padding_mask: Tensor


def load_tiny_corpus() -> tuple[str, ...]:
    """Load the packaged, line-oriented educational corpus."""
    resource = files("tiny_transformer").joinpath("data/tiny_corpus.txt")
    documents = tuple(
        line.strip()
        for line in resource.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(documents) < 2:
        raise ValueError("the tiny corpus must contain at least two documents")
    return documents


def split_corpus(
    documents: Sequence[str],
    *,
    validation_fraction: float = 0.2,
    seed: int = 17,
) -> CorpusSplit:
    """Split whole documents using a local generator without touching global RNG."""
    if len(documents) < 2:
        raise ValueError("documents must contain at least two items")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    validation_size = max(1, round(len(documents) * validation_fraction))
    validation_size = min(validation_size, len(documents) - 1)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(documents), generator=generator)
    validation_indexes = set(order[:validation_size].tolist())
    validation = tuple(
        document
        for index, document in enumerate(documents)
        if index in validation_indexes
    )
    train = tuple(
        document
        for index, document in enumerate(documents)
        if index not in validation_indexes
    )
    return CorpusSplit(train=train, validation=validation)


def build_next_token_batch(
    documents: Sequence[str],
    vocabulary: Vocabulary,
    *,
    sequence_length: int,
    stride: int | None = None,
) -> NextTokenBatch:
    """Create shifted examples without allowing sequences to cross documents."""
    if not documents:
        raise ValueError("documents must contain at least one item")
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")
    if stride is None:
        stride = sequence_length
    if stride < 1:
        raise ValueError("stride must be at least 1")

    input_rows: list[list[int]] = []
    target_rows: list[list[int]] = []
    mask_rows: list[list[bool]] = []
    for document in documents:
        token_ids = vocabulary.encode(document)
        for start in range(0, len(token_ids) - 1, stride):
            inputs = token_ids[start : start + sequence_length]
            targets = token_ids[start + 1 : start + sequence_length + 1]
            width = len(inputs)
            padding = sequence_length - width
            input_rows.append(inputs + [vocabulary.pad_id] * padding)
            target_rows.append(targets + [-100] * (sequence_length - len(targets)))
            mask_rows.append([True] * width + [False] * padding)

    if not input_rows:
        raise ValueError("documents must contain at least one adjacent token pair")
    return NextTokenBatch(
        input_ids=torch.tensor(input_rows, dtype=torch.long),
        target_ids=torch.tensor(target_rows, dtype=torch.long),
        padding_mask=torch.tensor(mask_rows, dtype=torch.bool),
    )
