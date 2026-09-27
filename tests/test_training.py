import pytest
import torch

from tiny_transformer.corpus import NextTokenBatch, build_next_token_batch
from tiny_transformer.tokenizer import Vocabulary
from tiny_transformer.training import (
    TrainingConfig,
    next_token_loss,
    train_language_model,
)


def build_training_inputs() -> tuple[Vocabulary, NextTokenBatch, NextTokenBatch]:
    train_documents = ("red blue red blue", "blue red blue red")
    validation_documents = ("red blue red",)
    vocabulary = Vocabulary.build((*train_documents, *validation_documents))
    train_batch = build_next_token_batch(
        train_documents,
        vocabulary,
        sequence_length=3,
        stride=1,
    )
    validation_batch = build_next_token_batch(
        validation_documents,
        vocabulary,
        sequence_length=3,
        stride=1,
    )
    return vocabulary, train_batch, validation_batch


def test_next_token_loss_ignores_padded_targets() -> None:
    logits = torch.tensor([[[3.0, 0.0], [100.0, -100.0]]])
    target_ids = torch.tensor([[0, -100]])

    loss = next_token_loss(logits, target_ids)

    expected = torch.nn.functional.cross_entropy(
        logits[:, :1].squeeze(0),
        torch.tensor([0]),
    )
    torch.testing.assert_close(loss, expected)


def test_training_overfits_a_repeated_two_token_pattern() -> None:
    vocabulary, train_batch, validation_batch = build_training_inputs()

    run = train_language_model(
        train_batch,
        validation_batch,
        vocab_size=len(vocabulary),
        config=TrainingConfig(steps=30, learning_rate=0.03, embedding_dim=8),
    )

    assert len(run.train_losses) == 31
    assert run.train_losses[-1] < run.train_losses[0] * 0.1
    assert run.validation_losses[-1] < run.validation_losses[0]


def test_training_is_repeatable_and_preserves_the_global_rng() -> None:
    vocabulary, train_batch, validation_batch = build_training_inputs()
    config = TrainingConfig(steps=3, seed=5, embedding_dim=8)
    torch.manual_seed(44)
    expected = torch.rand(1)
    torch.manual_seed(44)

    first = train_language_model(
        train_batch,
        validation_batch,
        vocab_size=len(vocabulary),
        config=config,
    )
    second = train_language_model(
        train_batch,
        validation_batch,
        vocab_size=len(vocabulary),
        config=config,
    )
    actual = torch.rand(1)

    assert first.train_losses == second.train_losses
    assert first.validation_losses == second.validation_losses
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"steps": 0}, "steps"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"weight_decay": -0.1}, "weight_decay"),
    ],
)
def test_invalid_training_configuration_fails_clearly(
    kwargs: dict[str, int | float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TrainingConfig(**kwargs)


def test_loss_rejects_mismatched_targets() -> None:
    with pytest.raises(ValueError, match="first two"):
        next_token_loss(torch.randn(2, 3, 4), torch.ones((2, 2), dtype=torch.long))
    with pytest.raises(TypeError, match="torch.long"):
        next_token_loss(torch.randn(2, 3, 4), torch.ones((2, 3)))
