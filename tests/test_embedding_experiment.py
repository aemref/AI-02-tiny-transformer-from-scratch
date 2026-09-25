import torch

from tiny_transformer.experiments.embeddings import run_embedding_experiment


def test_embedding_experiment_measures_boundary_cases() -> None:
    experiment = run_embedding_experiment()
    summary = experiment.summary()

    assert summary["encoded_shape"] == [3, 4]
    assert summary["embedding_shape"] == [3, 4, 6]
    assert summary["unknown_token_count"] == 2
    assert summary["padding_token_count"] == 6
    assert summary["truncated_sequence_count"] == 1
    assert summary["padding_vector_l1"] == 0.0
    assert summary["token_gradient_rows"] == 4
    assert summary["position_gradient_rows"] == 4


def test_embedding_experiment_is_repeatable_without_changing_global_rng() -> None:
    torch.manual_seed(27)
    expected_next_value = torch.rand(1)
    torch.manual_seed(27)

    first = run_embedding_experiment(seed=5)
    second = run_embedding_experiment(seed=5)
    actual_next_value = torch.rand(1)

    torch.testing.assert_close(first.embeddings, second.embeddings)
    torch.testing.assert_close(actual_next_value, expected_next_value)
