import pytest
import torch

from tiny_transformer.generation import greedy_generate
from tiny_transformer.model import TinyTransformerLanguageModel
from tiny_transformer.tokenizer import Vocabulary


def build_model_and_vocabulary() -> tuple[TinyTransformerLanguageModel, Vocabulary]:
    vocabulary = Vocabulary.build(("red blue green",))
    torch.manual_seed(13)
    model = TinyTransformerLanguageModel(
        len(vocabulary),
        embedding_dim=6,
        max_sequence_length=2,
        num_heads=2,
    )
    return model, vocabulary


def test_greedy_generation_is_repeatable_and_uses_a_sliding_context() -> None:
    model, vocabulary = build_model_and_vocabulary()

    first = greedy_generate(
        model,
        vocabulary,
        prompt="red blue green",
        max_new_tokens=4,
    )
    second = greedy_generate(
        model,
        vocabulary,
        prompt="red blue green",
        max_new_tokens=4,
    )

    assert first == second
    assert first.prompt_tokens == ("red", "blue", "green")
    assert len(first.generated_tokens) == 4
    assert first.context_window == 2
    assert not {Vocabulary.PAD_TOKEN, Vocabulary.UNK_TOKEN}.intersection(
        first.generated_tokens
    )


def test_generation_restores_the_models_training_mode() -> None:
    model, vocabulary = build_model_and_vocabulary()
    model.train()

    greedy_generate(model, vocabulary, prompt="red", max_new_tokens=1)

    assert model.training


@pytest.mark.parametrize(
    ("prompt", "max_new_tokens", "message"),
    [
        ("   ", 1, "prompt"),
        ("red", 0, "max_new_tokens"),
    ],
)
def test_generation_rejects_invalid_requests(
    prompt: str, max_new_tokens: int, message: str
) -> None:
    model, vocabulary = build_model_and_vocabulary()

    with pytest.raises(ValueError, match=message):
        greedy_generate(
            model,
            vocabulary,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
