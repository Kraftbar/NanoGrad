"""Small metrics for demos and tests."""

from __future__ import annotations


def binary_accuracy(
    probabilities: list[float],
    targets: list[float],
    *,
    threshold: float = 0.5,
) -> float:
    """Return the fraction of binary predictions that match targets."""

    if len(probabilities) != len(targets):
        raise ValueError("probabilities and targets must have the same length")

    correct = 0
    for probability, target in zip(probabilities, targets):
        prediction = 1.0 if probability >= threshold else 0.0
        correct += prediction == target

    return correct / len(targets)
