import torch

from tiny_transformer.experiments.tensors import build_tensor_experiment


def test_tensor_experiment_preserves_expected_shape_and_dtype() -> None:
    experiment = build_tensor_experiment()

    assert experiment.inputs.shape == (2, 3)
    assert experiment.scaled.shape == experiment.inputs.shape
    assert experiment.scaled.dtype == torch.float32


def test_column_scale_broadcasts_across_both_rows() -> None:
    experiment = build_tensor_experiment()

    torch.testing.assert_close(
        experiment.scaled,
        experiment.inputs * experiment.scale.unsqueeze(0),
    )


def test_local_generator_makes_the_experiment_repeatable() -> None:
    first = build_tensor_experiment(seed=19)
    second = build_tensor_experiment(seed=19)

    torch.testing.assert_close(first.inputs, second.inputs)
    assert first.inputs.data_ptr() != second.inputs.data_ptr()

