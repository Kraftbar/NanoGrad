"""Character-level text helpers for tiny language-model experiments."""

from __future__ import annotations

from collections.abc import Iterable

from datasets import TinyDataset


class CharVocab:
    """Stable character vocabulary with one-hot encoding helpers."""

    def __init__(self, chars: Iterable[str]) -> None:
        unique_chars = sorted(set(chars))
        if not unique_chars:
            raise ValueError("character vocabulary must not be empty")
        if any(len(char) != 1 for char in unique_chars):
            raise ValueError("character vocabulary entries must be single characters")

        self.itos = unique_chars
        self.stoi = {
            char: index
            for index, char in enumerate(self.itos)
        }

    @classmethod
    def from_text(cls, text: str) -> "CharVocab":
        return cls(text)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        indices = []
        for char in text:
            if char not in self.stoi:
                raise ValueError(f"unknown character: {char!r}")
            indices.append(self.stoi[char])
        return indices

    def decode(self, indices: Iterable[int]) -> str:
        chars = []
        for index in indices:
            if index < 0 or index >= len(self.itos):
                raise ValueError(f"unknown character index: {index}")
            chars.append(self.itos[index])
        return "".join(chars)

    def one_hot(self, index: int) -> list[float]:
        if index < 0 or index >= len(self.itos):
            raise ValueError(f"unknown character index: {index}")
        values = [0.0] * len(self.itos)
        values[index] = 1.0
        return values


def next_char_dataset(
    text: str,
    *,
    vocab: CharVocab | None = None,
    context_size: int = 1,
) -> tuple[TinyDataset, CharVocab]:
    """Build a flattened one-hot next-character prediction dataset."""

    if context_size <= 0:
        raise ValueError("context_size must be positive")
    if len(text) <= context_size:
        raise ValueError("text must be longer than context_size")

    vocab = vocab or CharVocab.from_text(text)
    encoded = vocab.encode(text)
    xs = []
    ys = []
    for start in range(len(encoded) - context_size):
        context = encoded[start : start + context_size]
        target = encoded[start + context_size]
        xs.append(_flatten_one_hot_context(context, vocab))
        ys.append(float(target))

    return TinyDataset(xs, ys), vocab


def next_char_index_dataset(
    text: str,
    *,
    vocab: CharVocab | None = None,
    context_size: int = 1,
) -> tuple[TinyDataset, CharVocab]:
    """Build a next-character prediction dataset with integer id contexts."""

    if context_size <= 0:
        raise ValueError("context_size must be positive")
    if len(text) <= context_size:
        raise ValueError("text must be longer than context_size")

    vocab = vocab or CharVocab.from_text(text)
    encoded = vocab.encode(text)
    xs = []
    ys = []
    for start in range(len(encoded) - context_size):
        xs.append([
            float(index)
            for index in encoded[start : start + context_size]
        ])
        ys.append(float(encoded[start + context_size]))

    return TinyDataset(xs, ys), vocab


def _flatten_one_hot_context(context: list[int], vocab: CharVocab) -> list[float]:
    values = []
    for index in context:
        values.extend(vocab.one_hot(index))
    return values
