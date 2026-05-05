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


def tensor_multiclass_accuracy(logits: Tensor, targets: Tensor | list[int]) -> float:
    """Return accuracy from 2D class logits and integer class targets."""

    if len(logits.shape) != 2:
        raise ValueError("multiclass accuracy expects 2D logits")

    rows, cols = logits.shape
    classes = _class_indices(targets, rows, cols)

    correct = 0
    for row, target in enumerate(classes):
        start = row * cols
        row_logits = logits.data[start : start + cols]
        prediction = max(range(cols), key=lambda col: row_logits[col])
        correct += prediction == target

    return correct / rows


def tensor_multiclass_confusion_matrix(
    logits: Tensor,
    targets: Tensor | list[int],
) -> list[list[int]]:
    """Return counts indexed by true class row and predicted class column."""

    if len(logits.shape) != 2:
        raise ValueError("multiclass confusion matrix expects 2D logits")

    rows, cols = logits.shape
    classes = _class_indices(targets, rows, cols)
    matrix = [
        [0 for _ in range(cols)]
        for _ in range(cols)
    ]

    for row, target in enumerate(classes):
        start = row * cols
        row_logits = logits.data[start : start + cols]
        prediction = max(range(cols), key=lambda col: row_logits[col])
        matrix[target][prediction] += 1

    return matrix


def _class_indices(targets: Tensor | list[int], rows: int, cols: int) -> list[int]:
    if isinstance(targets, Tensor):
        if targets.shape not in ((rows,), (rows, 1)):
            raise ValueError("class targets must have shape (batch,) or (batch, 1)")
        values = targets.data
    else:
        values = targets

    if len(values) != rows:
        raise ValueError("class targets length must match logits batch size")

    classes = []
    for value in values:
        index = int(value)
        if index != value or index < 0 or index >= cols:
            raise ValueError("class targets must be integer class indices")
        classes.append(index)
    return classes
