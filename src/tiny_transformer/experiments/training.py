"""Reproducible overfitting, regularization, and learning-rate experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from tiny_transformer.corpus import (
    build_next_token_batch,
    load_tiny_corpus,
    split_corpus,
)
from tiny_transformer.tokenizer import Vocabulary
from tiny_transformer.training import TrainingConfig, TrainingRun, train_language_model


@dataclass(frozen=True)
class ExperimentResult:
    """Named training run with JSON-compatible measured output."""

    name: str
    run: TrainingRun

    def summary(self) -> dict[str, object]:
        """Return configuration, full curves, and observed extrema."""
        validation_losses = self.run.validation_losses
        return {
            "name": self.name,
            "config": asdict(self.run.config),
            "initial_train_loss": self.run.train_losses[0],
            "final_train_loss": self.run.train_losses[-1],
            "initial_validation_loss": validation_losses[0],
            "final_validation_loss": validation_losses[-1],
            "best_validation_loss": min(validation_losses),
            "best_validation_step": validation_losses.index(min(validation_losses)),
            "train_loss_curve": list(self.run.train_losses),
            "validation_loss_curve": list(validation_losses),
        }


def run_training_experiments(*, steps: int = 60) -> dict[str, object]:
    """Run three controlled configurations on one fixed corpus split."""
    documents = load_tiny_corpus()
    split = split_corpus(documents, validation_fraction=0.25, seed=19)
    vocabulary = Vocabulary.build(documents)
    experiment_train_documents = split.train[:6]
    train_batch = build_next_token_batch(
        experiment_train_documents,
        vocabulary,
        sequence_length=6,
        stride=3,
    )
    validation_batch = build_next_token_batch(
        split.validation,
        vocabulary,
        sequence_length=6,
        stride=3,
    )
    configurations = (
        (
            "unregularized_overfit",
            TrainingConfig(steps=steps, learning_rate=0.03, seed=29),
        ),
        (
            "dropout_and_weight_decay",
            TrainingConfig(
                steps=steps,
                learning_rate=0.03,
                dropout=0.2,
                weight_decay=0.02,
                seed=29,
            ),
        ),
        (
            "lower_learning_rate",
            TrainingConfig(steps=steps, learning_rate=0.003, seed=29),
        ),
    )
    results = tuple(
        ExperimentResult(
            name=name,
            run=train_language_model(
                train_batch,
                validation_batch,
                vocab_size=len(vocabulary),
                config=config,
            ),
        )
        for name, config in configurations
    )
    return {
        "corpus_document_count": len(documents),
        "experiment_train_document_count": len(experiment_train_documents),
        "validation_document_count": len(split.validation),
        "vocabulary_size": len(vocabulary),
        "sequence_length": train_batch.input_ids.shape[1],
        "train_example_count": train_batch.input_ids.shape[0],
        "validation_example_count": validation_batch.input_ids.shape[0],
        "experiments": [result.summary() for result in results],
    }


def main() -> None:
    """Print all measured configurations and loss curves as JSON."""
    print(json.dumps(run_training_experiments(), indent=2))


if __name__ == "__main__":
    main()
