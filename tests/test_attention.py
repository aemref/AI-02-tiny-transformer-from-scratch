import pytest
import torch

from tiny_transformer.attention import scaled_dot_product_attention


def test_scaled_dot_product_attention_matches_manual_calculation() -> None:
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
    key = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
    value = torch.tensor([[[2.0, 1.0], [0.0, 3.0]]], requires_grad=True)

    output = scaled_dot_product_attention(query, key, value)
    expected_weights = torch.softmax((query @ key.transpose(-2, -1)) / 2**0.5, -1)

    torch.testing.assert_close(output.weights, expected_weights)
    torch.testing.assert_close(output.values, expected_weights @ value)
    torch.testing.assert_close(output.weights.sum(dim=-1), torch.ones((1, 2)))


def test_attention_supports_different_query_and_value_dimensions() -> None:
    query = torch.randn(2, 3, 4)
    key = torch.randn(2, 5, 4)
    value = torch.randn(2, 5, 6)

    output = scaled_dot_product_attention(query, key, value)

    assert output.weights.shape == (2, 3, 5)
    assert output.values.shape == (2, 3, 6)


def test_attention_preserves_gradient_flow_to_every_input() -> None:
    query = torch.randn(2, 3, 4, requires_grad=True)
    key = torch.randn(2, 3, 4, requires_grad=True)
    value = torch.randn(2, 3, 5, requires_grad=True)

    scaled_dot_product_attention(query, key, value).values.square().sum().backward()

    for tensor in (query, key, value):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) > 0


@pytest.mark.parametrize(
    ("query_shape", "key_shape", "value_shape", "message"),
    [
        ((4,), (2, 4), (2, 3), "at least two dimensions"),
        ((2, 3, 4), (1, 3, 4), (1, 3, 5), "batch dimensions"),
        ((1, 3, 4), (1, 3, 5), (1, 3, 6), "feature dimensions"),
        ((1, 3, 4), (1, 2, 4), (1, 3, 6), "sequence lengths"),
    ],
)
def test_attention_rejects_incompatible_shapes(
    query_shape: tuple[int, ...],
    key_shape: tuple[int, ...],
    value_shape: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        scaled_dot_product_attention(
            torch.randn(query_shape),
            torch.randn(key_shape),
            torch.randn(value_shape),
        )


def test_attention_rejects_non_floating_queries() -> None:
    query = torch.ones((1, 2, 3), dtype=torch.long)
    key = torch.ones((1, 2, 3), dtype=torch.long)
    value = torch.ones((1, 2, 4), dtype=torch.long)

    with pytest.raises(TypeError, match="floating-point"):
        scaled_dot_product_attention(query, key, value)
