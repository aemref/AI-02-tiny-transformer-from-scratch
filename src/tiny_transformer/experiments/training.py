"""Reproducible overfitting, regularization, and learning-rate experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

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


def render_loss_curves(summary: dict[str, object]) -> str:
    """Render dependency-free SVG small multiples from measured loss curves."""
    experiments = summary["experiments"]
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("summary must contain at least one experiment")
    width = 900
    panel_height = 230
    height = panel_height * len(experiments)
    plot_left, plot_right = 70, width - 30
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:ui-monospace,monospace;fill:#172033}'
        '.axis{stroke:#9aa5b5;stroke-width:1}.train{fill:none;stroke:#1769aa;'
        'stroke-width:2}.validation{fill:none;stroke:#d95f02;stroke-width:2;'
        'stroke-dasharray:6 4}</style>',
    ]
    for index, experiment in enumerate(experiments):
        train = experiment["train_loss_curve"]
        validation = experiment["validation_loss_curve"]
        max_loss = max((*train, *validation))
        top = index * panel_height + 42
        bottom = (index + 1) * panel_height - 38
        span = max(1, len(train) - 1)

        def points(
            curve: list[float],
            *,
            span: int = span,
            top: int = top,
            bottom: int = bottom,
            max_loss: float = max_loss,
        ) -> str:
            return " ".join(
                f"{plot_left + (plot_right - plot_left) * step / span:.1f},"
                f"{bottom - (bottom - top) * loss / max_loss:.1f}"
                for step, loss in enumerate(curve)
            )

        elements.extend(
            [
                f'<text x="20" y="{index * panel_height + 24}" '
                f'font-size="15" font-weight="600">{experiment["name"]}</text>',
                f'<line class="axis" x1="{plot_left}" y1="{top}" '
                f'x2="{plot_left}" y2="{bottom}"/>',
                f'<line class="axis" x1="{plot_left}" y1="{bottom}" '
                f'x2="{plot_right}" y2="{bottom}"/>',
                f'<text x="8" y="{top + 5}" font-size="11">{max_loss:.2f}</text>',
                f'<text x="45" y="{bottom + 4}" font-size="11">0</text>',
                f'<text x="{plot_left}" y="{bottom + 20}" font-size="11">0</text>',
                f'<text x="{plot_right - 20}" y="{bottom + 20}" '
                f'font-size="11">{span}</text>',
                f'<polyline class="train" points="{points(train)}"/>',
                f'<polyline class="validation" points="{points(validation)}"/>',
            ]
        )
    elements.extend(
        [
            '<line class="train" x1="660" y1="18" x2="690" y2="18"/>',
            '<text x="698" y="22" font-size="11">train</text>',
            '<line class="validation" x1="760" y1="18" x2="790" y2="18"/>',
            '<text x="798" y="22" font-size="11">validation</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements)


def write_experiment_artifacts(
    summary: dict[str, object],
    output_directory: Path,
) -> tuple[Path, Path]:
    """Write measured JSON and its SVG visualization to an explicit directory."""
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "training-results.json"
    svg_path = output_directory / "training-loss-curves.svg"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    svg_path.write_text(render_loss_curves(summary) + "\n", encoding="utf-8")
    return json_path, svg_path


def main() -> None:
    """Run experiments, printing JSON or writing JSON and SVG artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write training-results.json and training-loss-curves.svg here",
    )
    args = parser.parse_args()
    summary = run_training_experiments()
    if args.output_dir is None:
        print(json.dumps(summary, indent=2))
        return
    paths = write_experiment_artifacts(summary, args.output_dir)
    print(json.dumps({"artifacts": [str(path) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
