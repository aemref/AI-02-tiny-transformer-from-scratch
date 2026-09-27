import json
from pathlib import Path

import torch

from tiny_transformer.experiments.training import (
    render_loss_curves,
    run_training_experiments,
    write_experiment_artifacts,
)


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


def test_loss_curve_renderer_draws_each_measured_series() -> None:
    summary = run_training_experiments(steps=1)

    svg = render_loss_curves(summary)

    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert svg.count('class="train" points=') == 3
    assert svg.count('class="validation" points=') == 3
    assert "unregularized_overfit" in svg


def test_artifact_writer_persists_json_and_svg(
    tmp_path: Path,
) -> None:
    summary = run_training_experiments(steps=1)

    json_path, svg_path = write_experiment_artifacts(summary, tmp_path / "evidence")

    assert json.loads(json_path.read_text()) == summary
    assert svg_path.read_text().endswith("</svg>\n")
