import pytest
import torch

from tiny_transformer.embeddings import TokenEmbedding


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
