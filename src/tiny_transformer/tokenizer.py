"""A deterministic whitespace tokenizer and explicit token vocabulary."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


def tokenize(text: str) -> list[str]:
    """Lowercase ``text`` and split it on runs of whitespace."""
    return text.lower().split()


@dataclass(frozen=True)
class Vocabulary:
    """A token-to-integer mapping with fixed padding and unknown tokens."""

    tokens: tuple[str, ...]

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __post_init__(self) -> None:
        if len(self.tokens) < 2 or self.tokens[:2] != (
            self.PAD_TOKEN,
            self.UNK_TOKEN,
        ):
            raise ValueError("vocabulary must begin with <pad> and <unk>")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("vocabulary tokens must be unique")

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        *,
        min_frequency: int = 1,
        max_size: int | None = None,
    ) -> Vocabulary:
        """Build a repeatable vocabulary ordered by frequency, then token."""
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")
        if max_size is not None and max_size < 2:
            raise ValueError("max_size must leave room for special tokens")

        counts = Counter(token for text in texts for token in tokenize(text))
        candidates = sorted(
            (token for token, count in counts.items() if count >= min_frequency),
            key=lambda token: (-counts[token], token),
        )
        if max_size is not None:
            candidates = candidates[: max_size - 2]
        return cls((cls.PAD_TOKEN, cls.UNK_TOKEN, *candidates))

    @property
    def pad_id(self) -> int:
        """Return the reserved padding identifier."""
        return 0

    @property
    def unk_id(self) -> int:
        """Return the reserved unknown-token identifier."""
        return 1

    def __len__(self) -> int:
        return len(self.tokens)

    def encode_tokens(self, tokens: Sequence[str]) -> list[int]:
        """Map tokens to ids, using ``unk_id`` for unseen values."""
        token_to_id = {token: index for index, token in enumerate(self.tokens)}
        return [token_to_id.get(token.lower(), self.unk_id) for token in tokens]

    def encode(self, text: str) -> list[int]:
        """Tokenize and encode one text string."""
        return self.encode_tokens(tokenize(text))

    def decode(self, token_ids: Sequence[int]) -> list[str]:
        """Map valid token ids back to their vocabulary entries."""
        if any(token_id < 0 or token_id >= len(self) for token_id in token_ids):
            raise ValueError("token id is outside the vocabulary")
        return [self.tokens[token_id] for token_id in token_ids]
