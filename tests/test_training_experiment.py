import torch

from tiny_transformer.experiments.training import run_training_experiments


def test_training_experiment_reports_three_controlled_runs() -> None:
    summary = run_training_experiments(steps=2)
    experiments = summary["experiments"]

    assert summary["corpus_document_count"] == 24
    assert summary["experiment_train_document_count"] == 6
    assert summary["validation_document_count"] == 6
    assert summary["sequence_length"] == 6
    assert isinstance(experiments, list)
    assert [experiment["name"] for experiment in experiments] == [
        "unregularized_overfit",
        "dropout_and_weight_decay",
        "lower_learning_rate",
    ]
    assert all(len(experiment["train_loss_curve"]) == 3 for experiment in experiments)
    assert all(
        len(experiment["validation_loss_curve"]) == 3 for experiment in experiments
    )
    assert experiments[0]["config"]["dropout"] == 0.0
    assert experiments[1]["config"]["dropout"] == 0.2
    assert experiments[2]["config"]["learning_rate"] == 0.003


def test_training_experiment_is_repeatable_and_preserves_rng() -> None:
    torch.manual_seed(52)
    expected = torch.rand(1)
    torch.manual_seed(52)

    first = run_training_experiments(steps=1)
    second = run_training_experiments(steps=1)
    actual = torch.rand(1)

    assert first == second
    torch.testing.assert_close(actual, expected)
