import torch

from tiny_transformer.experiments.autograd import compute_weight_gradient


def test_backward_populates_the_expected_scalar_gradient() -> None:
    experiment = compute_weight_gradient(weight_value=0.5)
    manual_gradient = torch.mean(
        2 * (experiment.predictions - experiment.targets) * experiment.inputs
    )

    torch.testing.assert_close(experiment.weight_gradient, manual_gradient)
    assert experiment.weight_gradient.ndim == 0


def test_gradient_points_toward_lower_loss_for_gradient_descent() -> None:
    experiment = compute_weight_gradient(weight_value=0.5)
    learning_rate = 0.1
    updated_weight = experiment.weight - learning_rate * experiment.weight_gradient
    updated_errors = updated_weight * experiment.inputs - experiment.targets
    updated_loss = torch.mean(updated_errors**2)

    assert updated_loss < experiment.loss


def test_exact_solution_has_zero_gradient_and_loss() -> None:
    experiment = compute_weight_gradient(weight_value=2.0)

    torch.testing.assert_close(experiment.loss, torch.tensor(0.0, dtype=torch.float64))
    torch.testing.assert_close(
        experiment.weight_gradient,
        torch.tensor(0.0, dtype=torch.float64),
    )
