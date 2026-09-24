import pytest
import torch

from tiny_transformer.experiments.optimizers import fit_linear_model


def test_sgd_reduces_loss_and_recovers_the_linear_parameters() -> None:
    run = fit_linear_model()

    assert run.losses[-1] < run.losses[0] * 1e-4
    assert run.learned_weight == pytest.approx(3.0, abs=0.01)
    assert run.learned_bias == pytest.approx(-0.5, abs=0.001)


def test_zero_initialization_makes_the_training_run_repeatable() -> None:
    first = fit_linear_model(steps=20)
    second = fit_linear_model(steps=20)

    assert first.losses == second.losses
    assert first.learned_weight == second.learned_weight
    torch.testing.assert_close(first.final_gradient, second.final_gradient)


@pytest.mark.parametrize(
    ("steps", "learning_rate", "message"),
    [(0, 0.1, "steps must be at least 1"), (1, 0.0, "learning_rate must be positive")],
)
def test_invalid_optimizer_configuration_fails_clearly(
    steps: int,
    learning_rate: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_linear_model(steps=steps, learning_rate=learning_rate)

