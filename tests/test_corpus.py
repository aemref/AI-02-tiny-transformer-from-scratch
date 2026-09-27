import pytest
import torch

from tiny_transformer.corpus import (
    build_next_token_batch,
    load_tiny_corpus,
    split_corpus,
)
from tiny_transformer.tokenizer import Vocabulary


def test_packaged_corpus_is_nonempty_and_explicitly_licensed() -> None:
    documents = load_tiny_corpus()

    assert len(documents) == 24
    assert all(document.strip() == document for document in documents)
    assert all("\n" not in document for document in documents)


def test_split_is_repeatable_and_does_not_change_global_rng() -> None:
    documents = ("zero", "one", "two", "three", "four")
    torch.manual_seed(31)
    expected = torch.rand(1)
    torch.manual_seed(31)

    first = split_corpus(documents, validation_fraction=0.4, seed=8)
    second = split_corpus(documents, validation_fraction=0.4, seed=8)
    actual = torch.rand(1)

    assert first == second
    assert len(first.train) == 3
    assert len(first.validation) == 2
    assert set(first.train).isdisjoint(first.validation)
    assert set(first.train) | set(first.validation) == set(documents)
    torch.testing.assert_close(actual, expected)


def test_next_token_batch_shifts_targets_and_pads_short_sequences() -> None:
    documents = ("red blue green", "cat sat")
    vocabulary = Vocabulary.build(documents)

    batch = build_next_token_batch(documents, vocabulary, sequence_length=3)

    assert vocabulary.decode(batch.input_ids[0].tolist()) == ["red", "blue", "green"]
    assert batch.target_ids[0].tolist() == [
        vocabulary.encode("blue")[0],
        vocabulary.encode("green")[0],
        -100,
    ]
    assert batch.padding_mask.tolist() == [
        [True, True, True],
        [True, True, False],
    ]
    assert batch.input_ids[1, -1].item() == vocabulary.pad_id
    assert batch.target_ids[1, -1].item() == -100


def test_examples_never_cross_document_boundaries() -> None:
    documents = ("alpha omega", "first last")
    vocabulary = Vocabulary.build(documents)

    batch = build_next_token_batch(documents, vocabulary, sequence_length=4)

    first_targets = batch.target_ids[0][batch.target_ids[0].ne(-100)].tolist()
    second_targets = batch.target_ids[1][batch.target_ids[1].ne(-100)].tolist()
    assert vocabulary.decode(first_targets) == ["omega"]
    assert vocabulary.decode(second_targets) == ["last"]


@pytest.mark.parametrize(
    ("documents", "sequence_length", "stride", "message"),
    [
        ((), 2, None, "at least one"),
        (("two tokens",), 0, None, "sequence_length"),
        (("two tokens",), 2, 0, "stride"),
        (("alone",), 2, None, "adjacent token pair"),
    ],
)
def test_invalid_batch_inputs_fail_clearly(
    documents: tuple[str, ...],
    sequence_length: int,
    stride: int | None,
    message: str,
) -> None:
    vocabulary = Vocabulary.build(("two tokens",))

    with pytest.raises(ValueError, match=message):
        build_next_token_batch(
            documents,
            vocabulary,
            sequence_length=sequence_length,
            stride=stride,
        )


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1])
def test_invalid_split_fraction_fails_clearly(fraction: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        split_corpus(("one", "two"), validation_fraction=fraction)
