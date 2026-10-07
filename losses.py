"""Loss functions for tensor experiments."""

from __future__ import annotations

import math

import kernels
from tensor import Tensor, _add_grad, _grad_data, _op_result, _set_backward


def mse_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Mean squared error for same-shaped tensors."""

    _require_same_shape(predictions, targets)

    errors = predictions - targets
    return (errors * errors).mean()


def binary_cross_entropy(
    probabilities: Tensor,
    targets: Tensor,
    *,
    eps: float = 1e-12,
) -> Tensor:
    """Binary cross entropy for same-shaped probability tensors."""

    _require_same_shape(probabilities, targets)
    _require_binary_targets(targets)
    _require_probabilities(probabilities)

    losses = -(
        targets * (probabilities + eps).log()
        + (1 - targets) * (1 - probabilities + eps).log()
    )
    return losses.mean()


def softmax_cross_entropy(logits: Tensor, targets: Tensor | list[int]) -> Tensor:
    """Mean softmax cross entropy for 2D class logits."""

    if len(logits.shape) != 2:
        raise ValueError("softmax cross entropy expects 2D logits")

    rows, cols = logits.shape
    classes = _class_indices(targets, rows, cols)
    probabilities = kernels.softmax_rows(logits.data, cols)
    loss = -sum(math.log(probabilities[row * cols + target]) for row, target in enumerate(classes))
    out = _op_result([loss / rows], (1,), (logits,), "softmax_cross_entropy")

    def _backward() -> None:
        # d(loss)/d(logits) = (softmax - one_hot(target)) / rows
        grad = probabilities[:]
        for row, target in enumerate(classes):
            grad[row * cols + target] -= 1.0
        _add_grad(logits, kernels.scale(grad, _grad_data(out)[0] / rows))

    _set_backward(out, _backward)
    return out


def _require_same_shape(left: Tensor, right: Tensor) -> None:
    if left.shape != right.shape:
        raise ValueError("loss tensors must have the same shape")


def _require_binary_targets(targets: Tensor) -> None:
    if any(value not in (0.0, 1.0) for value in targets.data):
        raise ValueError("binary cross entropy targets must be 0 or 1")


def _require_probabilities(probabilities: Tensor) -> None:
    if any(value < 0.0 or value > 1.0 for value in probabilities.data):
        raise ValueError("binary cross entropy probabilities must be in [0, 1]")


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
