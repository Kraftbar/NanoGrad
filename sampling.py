"""Pick the next token from a row of logits: greedy, or top-k/temperature sampling.

Sampling recipe (the same one nanoGPT's ``generate`` uses):

1. keep only the ``top_k`` largest logits (optional)
2. divide by ``temperature``: < 1 sharpens, > 1 flattens the distribution
3. softmax into probabilities
4. draw one index with a uniform random number and the cumulative sum

Everything takes a plain list of floats so any model (NanoGrad or not) can be
sampled the same way with the same random stream.
"""

from __future__ import annotations

import math
import random


def argmax(values: list[float]) -> int:
    """Index of the largest value (first one on ties)."""

    if not values:
        raise ValueError("argmax values must not be empty")
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def sample_from_logits(
    values: list[float],
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    rng: random.Random | None = None,
) -> int:
    """Draw one index from softmax(values / temperature), limited to top_k."""

    if not values:
        raise ValueError("sample values must not be empty")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")

    generator = rng or random.Random()
    indexed_values = list(enumerate(values))
    if top_k is not None:
        indexed_values = sorted(
            indexed_values,
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

    row_max = max(value for _index, value in indexed_values)
    weights = [
        math.exp((value - row_max) / temperature)
        for _index, value in indexed_values
    ]
    total = sum(weights)
    threshold = generator.random() * total
    cumulative = 0.0
    for (index, _value), weight in zip(indexed_values, weights):
        cumulative += weight
        if threshold <= cumulative:
            return index
    return indexed_values[-1][0]


def select_next_index(
    values: list[float],
    *,
    sample_mode: str,
    temperature: float = 1.0,
    top_k: int | None = None,
    rng: random.Random | None = None,
) -> int:
    """Greedy argmax or a sampled index, chosen by ``sample_mode``."""

    if sample_mode == "greedy":
        return argmax(values)
    if sample_mode == "sample":
        return sample_from_logits(
            values,
            temperature=temperature,
            top_k=top_k,
            rng=rng,
        )
    raise ValueError(f"unknown sample mode: {sample_mode}")
