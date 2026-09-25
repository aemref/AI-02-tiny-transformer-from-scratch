import pytest
import torch

from tiny_transformer.embeddings import (
    PositionalEmbedding,
    TokenEmbedding,
    TransformerEmbedding,
)


def test_token_embedding_appends_feature_dimension() -> None:
    layer = TokenEmbedding(vocab_size=6, embedding_dim=4)
    token_ids = torch.tensor([[2, 3, 0], [4, 1, 5]], dtype=torch.long)

    output = layer(token_ids)

    assert output.shape == (2, 3, 4)


def test_padding_vector_is_zero_and_receives_no_gradient() -> None:
    layer = TokenEmbedding(vocab_size=5, embedding_dim=3, padding_idx=0)
    token_ids = torch.tensor([[2, 0, 3]], dtype=torch.long)

    output = layer(token_ids)
    output.sum().backward()

    torch.testing.assert_close(output[0, 1], torch.zeros(3))
    assert layer.weight.grad is not None
    torch.testing.assert_close(layer.weight.grad[0], torch.zeros(3))
    torch.testing.assert_close(layer.weight.grad[1], torch.zeros(3))
    assert torch.count_nonzero(layer.weight.grad[2]) == 3
    assert torch.count_nonzero(layer.weight.grad[3]) == 3


def test_repeated_token_accumulates_gradient_in_one_table_row() -> None:
    layer = TokenEmbedding(vocab_size=4, embedding_dim=2)
    token_ids = torch.tensor([[2, 2]], dtype=torch.long)

    layer(token_ids).sum().backward()

    assert layer.weight.grad is not None
    torch.testing.assert_close(layer.weight.grad[2], torch.tensor([2.0, 2.0]))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"vocab_size": 1, "embedding_dim": 2}, "at least two tokens"),
        ({"vocab_size": 3, "embedding_dim": 0}, "embedding_dim"),
        ({"vocab_size": 3, "embedding_dim": 2, "padding_idx": 3}, "padding_idx"),
    ],
)
def test_invalid_token_embedding_configuration_fails_clearly(
    kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TokenEmbedding(**kwargs)


def test_token_embedding_requires_rank_two_long_ids() -> None:
    layer = TokenEmbedding(vocab_size=4, embedding_dim=2)

    with pytest.raises(ValueError, match="batch x sequence"):
        layer(torch.tensor([1, 2], dtype=torch.long))
    with pytest.raises(TypeError, match="torch.long"):
        layer(torch.tensor([[1.0, 2.0]]))


def test_positional_embedding_broadcasts_across_batch() -> None:
    layer = PositionalEmbedding(max_sequence_length=4, embedding_dim=3)
    token_ids = torch.tensor([[2, 2, 2], [3, 3, 3]], dtype=torch.long)

    output = layer(token_ids)

    assert output.shape == (2, 3, 3)
    torch.testing.assert_close(output[0], output[1])
    assert not torch.equal(output[0, 0], output[0, 1])


def test_transformer_embedding_zeroes_padding_after_adding_positions() -> None:
    layer = TransformerEmbedding(
        vocab_size=5,
        embedding_dim=4,
        max_sequence_length=3,
    )
    token_ids = torch.tensor([[2, 0, 3]], dtype=torch.long)

    output = layer(token_ids)

    torch.testing.assert_close(output[0, 1], torch.zeros(4))
    assert torch.count_nonzero(output[0, 0]) == 4


def test_transformer_embedding_propagates_non_padding_gradients() -> None:
    layer = TransformerEmbedding(
        vocab_size=5,
        embedding_dim=2,
        max_sequence_length=4,
    )
    token_ids = torch.tensor([[2, 3, 0]], dtype=torch.long)

    layer(token_ids).sum().backward()

    assert layer.tokens.weight.grad is not None
    assert layer.positions.weight.grad is not None
    torch.testing.assert_close(layer.tokens.weight.grad[0], torch.zeros(2))
    torch.testing.assert_close(layer.positions.weight.grad[2], torch.zeros(2))
    torch.testing.assert_close(layer.positions.weight.grad[3], torch.zeros(2))
    assert torch.count_nonzero(layer.tokens.weight.grad[2]) == 2
    assert torch.count_nonzero(layer.positions.weight.grad[0]) == 2


def test_positional_embedding_rejects_sequence_beyond_limit() -> None:
    layer = PositionalEmbedding(max_sequence_length=2, embedding_dim=3)
    token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with pytest.raises(ValueError, match="exceeds configured maximum 2"):
        layer(token_ids)


@pytest.mark.parametrize(
    ("max_sequence_length", "embedding_dim", "message"),
    [(0, 2, "max_sequence_length"), (2, 0, "embedding_dim")],
)
def test_invalid_positional_embedding_configuration_fails_clearly(
    max_sequence_length: int, embedding_dim: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PositionalEmbedding(max_sequence_length, embedding_dim)
