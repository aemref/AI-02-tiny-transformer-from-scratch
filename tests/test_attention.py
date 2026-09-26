import pytest
import torch

from tiny_transformer.attention import (
    causal_attention_mask,
    scaled_dot_product_attention,
)


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


def test_boolean_mask_excludes_keys_and_renormalizes_visible_weights() -> None:
    query = torch.ones((1, 2, 2))
    key = torch.ones((1, 3, 2))
    value = torch.tensor([[[1.0], [10.0], [100.0]]])
    mask = torch.tensor([[True, False, True], [False, True, False]])

    output = scaled_dot_product_attention(
        query,
        key,
        value,
        attention_mask=mask,
    )

    torch.testing.assert_close(
        output.weights,
        torch.tensor([[[0.5, 0.0, 0.5], [0.0, 1.0, 0.0]]]),
    )
    torch.testing.assert_close(output.values, torch.tensor([[[50.5], [10.0]]]))


@pytest.mark.parametrize(
    ("mask", "exception", "message"),
    [
        (torch.ones((2, 2)), TypeError, "torch.bool"),
        (torch.ones((4, 4), dtype=torch.bool), ValueError, "broadcast"),
        (torch.zeros((2, 3), dtype=torch.bool), ValueError, "every key"),
    ],
)
def test_attention_rejects_invalid_masks(
    mask: torch.Tensor,
    exception: type[Exception],
    message: str,
) -> None:
    query = torch.ones((1, 2, 3))
    key = torch.ones((1, 3, 3))
    value = torch.ones((1, 3, 4))

    with pytest.raises(exception, match=message):
        scaled_dot_product_attention(
            query,
            key,
            value,
            attention_mask=mask,
        )


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


def test_causal_mask_has_the_expected_lower_triangular_structure() -> None:
    mask = causal_attention_mask(4)

    assert mask.dtype == torch.bool
    torch.testing.assert_close(
        mask,
        torch.tensor(
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        ),
    )


def test_causal_attention_prevents_future_values_from_changing_the_past() -> None:
    query = torch.ones((1, 3, 2))
    key = torch.ones((1, 3, 2))
    values = torch.tensor([[[1.0], [2.0], [3.0]]])
    changed_future = torch.tensor([[[1.0], [20.0], [30.0]]])
    mask = causal_attention_mask(3)

    baseline = scaled_dot_product_attention(
        query,
        key,
        values,
        attention_mask=mask,
    )
    changed = scaled_dot_product_attention(
        query,
        key,
        changed_future,
        attention_mask=mask,
    )

    torch.testing.assert_close(baseline.values[:, 0], changed.values[:, 0])
    assert baseline.weights[0, 0].tolist() == [1.0, 0.0, 0.0]


def test_causal_mask_rejects_empty_sequences() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        causal_attention_mask(0)
