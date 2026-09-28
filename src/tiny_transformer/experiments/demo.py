"""Train locally and print one deterministic, auditable generation example."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict

from tiny_transformer.corpus import (
    build_next_token_batch,
    load_tiny_corpus,
    split_corpus,
)
from tiny_transformer.generation import greedy_generate
from tiny_transformer.tokenizer import Vocabulary
from tiny_transformer.training import TrainingConfig, train_language_model


def run_demo(
    *,
    prompt: str = "attention uses",
    max_new_tokens: int = 6,
    steps: int = 60,
) -> dict[str, object]:
    """Train the fixed demo configuration and return its complete local evidence."""
    documents = load_tiny_corpus()
    split = split_corpus(documents, validation_fraction=0.25, seed=19)
    vocabulary = Vocabulary.build(documents)
    train_documents = split.train[:6]
    train_batch = build_next_token_batch(
        train_documents,
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
    config = TrainingConfig(steps=steps, learning_rate=0.003, seed=29)
    run = train_language_model(
        train_batch,
        validation_batch,
        vocab_size=len(vocabulary),
        config=config,
    )
    result = greedy_generate(
        run.model,
        vocabulary,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )
    return {
        "prompt": prompt,
        "prompt_tokens": list(result.prompt_tokens),
        "generated_tokens": list(result.generated_tokens),
        "context_window": result.context_window,
        "training": {
            "config": asdict(config),
            "steps": config.steps,
            "initial_loss": run.train_losses[0],
            "final_loss": run.train_losses[-1],
            "validation_loss": run.validation_losses[-1],
            "train_document_count": len(train_documents),
            "validation_document_count": len(split.validation),
            "vocabulary_size": len(vocabulary),
        },
        "limitations": (
            "Greedy output from a six-document training subset is a mechanics "
            "demo, not evidence of language quality or generalization."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Parse demo options and print a stable JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="attention uses")
    parser.add_argument("--tokens", type=int, default=6)
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run_demo(
                prompt=args.prompt,
                max_new_tokens=args.tokens,
                steps=args.steps,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
