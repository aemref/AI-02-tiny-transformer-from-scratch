import pytest
import torch

from tiny_transformer.model import TinyTransformerLanguageModel


def build_model(*, dropout: float = 0.0) -> TinyTransformerLanguageModel:
    torch.manual_seed(11)
    return TinyTransformerLanguageModel(
        vocab_size=12,
        embedding_dim=8,
        max_sequence_length=5,
        num_heads=2,
        num_layers=2,
        dropout=dropout,
    )


def test_language_model_returns_one_logit_per_token_and_vocabulary_item() -> None:
    model = build_model()
    token_ids = torch.tensor([[2, 3, 4], [5, 6, 0]])

    logits = model(token_ids)

    assert logits.shape == (2, 3, 12)
    torch.testing.assert_close(logits[1, 2], torch.zeros(12))


def test_causal_model_keeps_past_logits_independent_of_future_tokens() -> None:
    model = build_model().eval()
    baseline = torch.tensor([[2, 3, 4, 5]])
    changed_future = torch.tensor([[2, 3, 9, 10]])

    baseline_logits = model(baseline)
    changed_logits = model(changed_future)

    torch.testing.assert_close(baseline_logits[:, :2], changed_logits[:, :2])
    assert not torch.equal(baseline_logits[:, 2:], changed_logits[:, 2:])


def test_output_projection_shares_the_token_embedding_table() -> None:
    model = build_model()

    assert model.output.weight.data_ptr() == model.embeddings.tokens.weight.data_ptr()


def test_language_model_propagates_gradients_through_every_block() -> None:
    model = build_model()
    logits = model(torch.tensor([[2, 3, 4]]))

    logits.square().mean().backward()

    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_layers": 0}, "num_layers"),
        ({"dropout": -0.1}, "dropout"),
        ({"dropout": 1.0}, "dropout"),
        ({"feed_forward_dim": 0}, "feed_forward_dim"),
    ],
)
def test_invalid_model_configuration_fails_clearly(
    kwargs: dict[str, int | float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TinyTransformerLanguageModel(
            vocab_size=8,
            embedding_dim=4,
            max_sequence_length=3,
            num_heads=2,
            **kwargs,
        )


def test_invalid_model_inputs_fail_clearly() -> None:
    model = build_model()

    with pytest.raises(ValueError, match="batch x sequence"):
        model(torch.tensor([2, 3]))
    with pytest.raises(TypeError, match="torch.long"):
        model(torch.tensor([[2.0, 3.0]]))
    with pytest.raises(TypeError, match="torch.bool"):
        model(torch.tensor([[2, 3]]), padding_mask=torch.ones((1, 2)))
    with pytest.raises(ValueError, match="match token_ids"):
        model(
            torch.tensor([[2, 3]]),
            padding_mask=torch.ones((1, 3), dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="at least one real token"):
        model(
            torch.tensor([[0, 0]]),
            padding_mask=torch.zeros((1, 2), dtype=torch.bool),
        )
