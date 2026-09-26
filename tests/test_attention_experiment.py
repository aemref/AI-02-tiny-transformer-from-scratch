import torch

from tiny_transformer.experiments.attention import run_attention_experiment


def test_attention_experiment_produces_a_normalized_causal_map() -> None:
    experiment = run_attention_experiment()
    summary = experiment.summary()

    assert summary["tokens"] == ["the", "cat", "sat"]
    assert summary["representation_shape"] == [3, 2]
    assert summary["row_sums"] == [1.0, 1.0, 1.0]
    assert summary["future_attention_total"] == 0.0
    assert summary["most_attended_token_by_query"] == ["the", "cat", "cat"]
    assert len(summary["attention_entropy"]) == 3
    assert len(summary["context_vectors"]) == 3


def test_attention_experiment_is_exactly_repeatable() -> None:
    first = run_attention_experiment()
    second = run_attention_experiment()

    torch.testing.assert_close(first.attention_map, second.attention_map)
    torch.testing.assert_close(first.context_vectors, second.context_vectors)
