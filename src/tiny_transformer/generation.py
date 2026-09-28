"""Auditable greedy next-token generation for the tiny language model."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tiny_transformer.model import TinyTransformerLanguageModel
from tiny_transformer.tokenizer import Vocabulary, tokenize


@dataclass(frozen=True)
class GenerationResult:
    """Tokens emitted by one deterministic greedy decoding request."""

    prompt_tokens: tuple[str, ...]
    generated_tokens: tuple[str, ...]
    context_window: int


def greedy_generate(
    model: TinyTransformerLanguageModel,
    vocabulary: Vocabulary,
    *,
    prompt: str,
    max_new_tokens: int,
) -> GenerationResult:
    """Generate tokens greedily while excluding reserved vocabulary entries."""
    prompt_tokens = tuple(tokenize(prompt))
    if not prompt_tokens:
        raise ValueError("prompt must contain at least one token")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if len(vocabulary) <= 2:
        raise ValueError("vocabulary must contain at least one non-special token")
    if model.vocab_size != len(vocabulary):
        raise ValueError("model and vocabulary sizes must match")

    context_window = model.embeddings.positions.max_sequence_length
    token_ids = vocabulary.encode_tokens(prompt_tokens)
    generated_ids: list[int] = []
    parameter = next(model.parameters())
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                context = token_ids[-context_window:]
                inputs = torch.tensor(
                    [context],
                    dtype=torch.long,
                    device=parameter.device,
                )
                next_logits = model(inputs)[0, -1].clone()
                next_logits[:2] = -torch.inf
                next_id = int(next_logits.argmax().item())
                token_ids.append(next_id)
                generated_ids.append(next_id)
    finally:
        model.train(was_training)

    return GenerationResult(
        prompt_tokens=prompt_tokens,
        generated_tokens=tuple(vocabulary.decode(generated_ids)),
        context_window=context_window,
    )
