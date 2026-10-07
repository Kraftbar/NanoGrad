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


def binary_cross_entropy_with_logits(logits: Tensor, targets: Tensor, *, reduction: str = "mean") -> Tensor:
    """Binary cross entropy on raw scores x (logits), targets t in [0, 1].

    With p = sigmoid(x), -[t log p + (1 - t) log(1 - p)] simplifies to

        max(x, 0) - x * t + log(1 + e^-|x|)

    which never takes the log of 0 or overflows e^x, unlike computing p
    first. The gradient is simply sigmoid(x) - t. ``reduction`` is "mean"
    (over all values) or "sum".
    """

    _require_same_shape(logits, targets)
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be 'mean' or 'sum'")
    total = kernels.bce_with_logits_sum(logits.data, targets.data)
    scale = 1.0 / logits.numel if reduction == "mean" else 1.0
    out = _op_result([total * scale], (1,), (logits, targets), "bce_with_logits")

    def _backward() -> None:
        g = _grad_data(out)[0] * scale
        difference = kernels.add(kernels.sigmoid(logits.data), kernels.scale(targets.data, -1.0))
        if logits.requires_grad:
            _add_grad(logits, kernels.scale(difference, g))
        if targets.requires_grad:   # d/dt = -x
            _add_grad(targets, kernels.scale(logits.data, -g))

    _set_backward(out, _backward)
    return out


def softmax_cross_entropy(
    logits: Tensor, targets: Tensor | list[int], *, label_smoothing: float = 0.0,
) -> Tensor:
    """Mean softmax cross entropy for 2D class logits.

    With label smoothing e, the target is no longer all-or-nothing: the true
    class gets 1 - e and every class (including it) shares e equally. Per
    row, with p = softmax(x) and C classes:

        loss = (1 - e) * -log p_target + e * (logsumexp(x) - mean(x))
        dloss/dx = p - (1 - e) * one_hot(target) - e / C
    """

    if len(logits.shape) != 2:
        raise ValueError("softmax cross entropy expects 2D logits")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")

    rows, cols = logits.shape
    classes = _class_indices(targets, rows, cols)
    probabilities = kernels.softmax_rows(logits.data, cols)
    loss = -sum(math.log(probabilities[row * cols + target]) for row, target in enumerate(classes))
    if label_smoothing:
        # logsumexp(x) = x_target - log p_target, so the smooth part needs only row sums.
        x = logits.data
        smooth = sum(
            x[row * cols + target] - math.log(probabilities[row * cols + target])
            - sum(x[row * cols : (row + 1) * cols]) / cols
            for row, target in enumerate(classes)
        )
        loss = (1.0 - label_smoothing) * loss + label_smoothing * smooth
    out = _op_result([loss / rows], (1,), (logits,), "softmax_cross_entropy")

    def _backward() -> None:
        # d(loss)/d(logits) = (softmax - smoothed one_hot(target)) / rows
        grad = probabilities[:]
        for row, target in enumerate(classes):
            grad[row * cols + target] -= 1.0 - label_smoothing
        if label_smoothing:
            grad = kernels.add(grad, kernels.storage([-label_smoothing / cols]) * len(grad))
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
