import pytest
import torch

from tiny_transformer.tokenizer import Vocabulary, tokenize


def test_tokenizer_normalizes_case_and_whitespace() -> None:
    assert tokenize("  Tiny\nTransformer  TINY ") == ["tiny", "transformer", "tiny"]


def test_vocabulary_is_frequency_sorted_with_stable_ties() -> None:
    vocabulary = Vocabulary.build(["blue red blue", "green red"])

    assert vocabulary.tokens == ("<pad>", "<unk>", "blue", "red", "green")
    assert vocabulary.pad_id == 0
    assert vocabulary.unk_id == 1


def test_vocabulary_build_is_independent_of_document_order() -> None:
    first = Vocabulary.build(["one two", "two three"])
    second = Vocabulary.build(["two three", "one two"])

    assert first == second


def test_encode_uses_unknown_id_and_decode_exposes_special_token() -> None:
    vocabulary = Vocabulary.build(["known tokens"])
    encoded = vocabulary.encode("KNOWN missing")

    assert encoded == [2, vocabulary.unk_id]
    assert vocabulary.decode(encoded) == ["known", Vocabulary.UNK_TOKEN]


def test_frequency_and_size_limits_are_explicit() -> None:
    vocabulary = Vocabulary.build(
        ["keep keep also also drop"], min_frequency=2, max_size=3
    )

    assert vocabulary.tokens == ("<pad>", "<unk>", "also")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_frequency": 0}, "min_frequency must be at least 1"),
        ({"max_size": 1}, "max_size must leave room for special tokens"),
    ],
)
def test_invalid_vocabulary_limits_fail_clearly(
    kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Vocabulary.build(["text"], **kwargs)


@pytest.mark.parametrize("token_id", [-1, 3])
def test_decode_rejects_out_of_range_token_id(token_id: int) -> None:
    vocabulary = Vocabulary.build(["text"])

    with pytest.raises(ValueError, match="outside the vocabulary"):
        vocabulary.decode([token_id])


def test_batch_encoding_right_pads_and_marks_real_tokens() -> None:
    vocabulary = Vocabulary.build(["small batch example"])
    batch = vocabulary.encode_batch(["small example", "batch"], max_length=3)

    assert batch.token_ids.tolist() == [[4, 3, 0], [2, 0, 0]]
    assert batch.attention_mask.tolist() == [
        [True, True, False],
        [True, False, False],
    ]
    assert batch.original_lengths == (2, 1)
    assert batch.token_ids.dtype == torch.long
    assert batch.attention_mask.dtype == torch.bool


def test_batch_encoding_truncates_but_preserves_original_length() -> None:
    vocabulary = Vocabulary.build(["one two three four"])
    batch = vocabulary.encode_batch(["one two three four"], max_length=2)

    assert batch.token_ids.shape == (1, 2)
    assert batch.attention_mask.all()
    assert batch.original_lengths == (4,)


@pytest.mark.parametrize(
    ("texts", "max_length", "message"),
    [
        ([], 2, "texts must contain at least one item"),
        (["text"], 0, "max_length must be at least 1"),
    ],
)
def test_invalid_batch_contract_fails_clearly(
    texts: list[str], max_length: int, message: str
) -> None:
    vocabulary = Vocabulary.build(["text"])

    with pytest.raises(ValueError, match=message):
        vocabulary.encode_batch(texts, max_length=max_length)
