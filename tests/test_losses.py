import torch
import torch.nn.functional as F

from tiny_transformer.experiments.losses import evaluate_cross_entropy


def test_cross_entropy_matches_negative_target_log_probability() -> None:
    experiment = evaluate_cross_entropy()
    log_probabilities = F.log_softmax(experiment.logits, dim=-1)
    row_indexes = torch.arange(experiment.targets.shape[0])
    manual_loss = -log_probabilities[row_indexes, experiment.targets]

    torch.testing.assert_close(experiment.per_example_loss, manual_loss)
    torch.testing.assert_close(experiment.mean_loss, manual_loss.mean())


def test_cross_entropy_is_invariant_to_a_per_row_logit_shift() -> None:
    baseline = evaluate_cross_entropy()
    shifted_logits = baseline.logits + torch.tensor([[100.0], [-80.0]])
    shifted = evaluate_cross_entropy(shifted_logits, baseline.targets)

    torch.testing.assert_close(shifted.per_example_loss, baseline.per_example_loss)


def test_confident_wrong_logits_increase_loss() -> None:
    baseline = evaluate_cross_entropy()
    wrong_logits = baseline.logits.flip(dims=(1,))
    wrong = evaluate_cross_entropy(wrong_logits, baseline.targets)

    assert wrong.mean_loss > baseline.mean_loss

