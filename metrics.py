"""Small metrics for demos and tests."""

from __future__ import annotations

from tensor import Tensor


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


def tensor_binary_accuracy(
    probabilities: Tensor,
    targets: Tensor,
    *,
    threshold: float = 0.5,
) -> float:
    """Return binary accuracy for same-shaped probability tensors."""

    if probabilities.shape != targets.shape:
        raise ValueError("probabilities and targets must have the same shape")
    if any(target not in (0.0, 1.0) for target in targets.data):
        raise ValueError("binary accuracy targets must be 0 or 1")

    return binary_accuracy(
        probabilities=probabilities.data,
        targets=targets.data,
        threshold=threshold,
    )
