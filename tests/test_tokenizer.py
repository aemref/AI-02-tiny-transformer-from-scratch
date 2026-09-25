import pytest

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
