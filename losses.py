"""Loss functions for tensor experiments."""

from __future__ import annotations

import math

import kernels
from tensor import Tensor, _add_grad, _grad_data, _op_result, _set_backward, is_grad_enabled


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
        if logits.requires_grad:
            _add_grad(logits, kernels.bce_with_logits_backward(logits.data, targets.data, g))
        if targets.requires_grad:   # d/dt = -x
            _add_grad(targets, kernels.scale(logits.data, -g))

    _set_backward(out, _backward)
    return out


def softmax_cross_entropy(
    logits: Tensor, targets: Tensor | list[int], *, label_smoothing: float = 0.0,
    class_weights: list[float] | None = None,
) -> Tensor:
    """Mean softmax cross entropy for 2D class logits.

    ``class_weights`` (one per class) makes each row count w[target] times,
    and divides by the sum of those weights instead of the number of rows
    (PyTorch's ``weight=``): rare classes can count as much as common ones.

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
    if class_weights is not None:
        if label_smoothing:
            raise ValueError("class_weights and label_smoothing cannot be combined")
        if len(class_weights) != cols:
            raise ValueError("class_weights needs one weight per class")
        row_weights = [class_weights[target] for target in classes]
        total = sum(row_weights)
        loss = sum(w * _target_nll(logits.data, probabilities, row, cols, target)
                   for row, (w, target) in enumerate(zip(row_weights, classes)))
        out = _op_result([loss / total], (1,), (logits,), "softmax_cross_entropy")

        def _weighted_backward() -> None:
            # d(loss)/d(logits) row r = w_r (softmax - one_hot(target)) / sum of w
            grad = probabilities[:]
            for row, target in enumerate(classes):
                grad[row * cols + target] -= 1.0
            scale = kernels.storage([w / total for w in row_weights for _ in range(cols)])
            _add_grad(logits, kernels.scale(kernels.multiply(grad, scale), _grad_data(out)[0]))

        _set_backward(out, _weighted_backward)
        return out
    loss = sum(_target_nll(logits.data, probabilities, row, cols, target) for row, target in enumerate(classes))
    if label_smoothing:
        # logsumexp(x) = x_target - log p_target, so the smooth part needs only row sums.
        x = logits.data
        smooth = sum(
            x[row * cols + target] + _target_nll(x, probabilities, row, cols, target)
            - sum(x[row * cols : (row + 1) * cols]) / cols
            for row, target in enumerate(classes)
        )
        loss = (1.0 - label_smoothing) * loss + label_smoothing * smooth
    out = _op_result([loss / rows], (1,), (logits,), "softmax_cross_entropy")

    def _backward() -> None:
        # d(loss)/d(logits) = (softmax - smoothed one_hot(target)) / rows, times the
        # output's gradient: scale the softmax (one pass, a new array), then fix the
        # targets (one scatter-add of -(1 - smoothing) * factor at each row's target).
        factor = _grad_data(out)[0] / rows
        grad = kernels.scale(probabilities, factor)
        targets = kernels.index_table([row * cols + target for row, target in enumerate(classes)])
        kernels.scatter_add_indexed(kernels.storage([-(1.0 - label_smoothing) * factor] * rows), targets,
                                    batches=1, image_size=rows * cols, into=grad)
        if label_smoothing:
            grad = kernels.add(grad, kernels.storage([-label_smoothing * factor / cols] * len(grad)))
        _add_grad(logits, grad)

    _set_backward(out, _backward)
    return out


# linear_cross_entropy's chunks: as many rows as fit in CHUNK_BYTES of
# logits, in multiples of CHUNK_STEP rows. CHUNK_STEP is the block of rows
# kernels.c's matrix product sums over (K_BLOCK), so the weight gradient,
# added chunk by chunk, is summed in the same order as one product over
# every row. Fewer, larger chunks are faster: every chunk passes over the
# whole weight matrix and its gradient again (WikiText-2's 700 x 33,278
# output layer, float64, 8 threads: 274 ms unfused; in chunks of 256 rows
# 245 ms; in one chunk 203 ms).
CHUNK_BYTES = 256 << 20
CHUNK_STEP = 256


def linear_cross_entropy(
    inputs: Tensor, weight: Tensor, bias: Tensor, targets: Tensor | list[int], *, chunk_rows: int | None = None,
) -> Tensor:
    """``softmax_cross_entropy(linear(inputs, weight, bias), targets)``,
    without ever holding the whole logits matrix.

    A language model's output layer turns each of N positions into V word
    scores (WikiText-2: V = 33,278), and the loss only needs their softmax.
    Built whole, the logits are N x V numbers (186 MB for 700 positions in
    float64), and the unfused path makes three such arrays: the logits,
    their softmax, and its gradient. Here the rows go through in chunks
    (``chunk_rows``; by default as many as fit in 256 MB of logits), with
    one chunk's logits in memory at once:

        logits = x_chunk @ W^T + b
        grad   = (softmax(logits) - one_hot(target)) / N     (in place: kernels.softmax_xent_rows)
        dx_chunk = grad @ W,   dW += grad^T @ x_chunk,   db += grad summed over rows

    The loss is the last op, so its gradient with respect to the logits is
    known as soon as their softmax is: each chunk's share of dx, dW and db
    is computed during the forward pass, and backward only hands them on.
    The operations and their order are those of ``linear`` followed by
    ``softmax_cross_entropy``, so the results are the same bit for bit.
    (That holds when the loss's own gradient is 1, as in ``loss.backward()``;
    any other value scales the finished gradients, which can differ in the
    last bit.) Label smoothing and class weights are not supported here.

    Shapes: inputs (..., in), weight (V, in), bias (V,), one target per row.
    """

    vocab, features = weight.shape
    if len(bias.shape) != 1 or bias.shape[0] != vocab:
        raise ValueError("bias shape must match the weight's rows")
    if inputs.shape[-1] != features:
        raise ValueError("linear input features must match weight columns")
    if chunk_rows is None:
        row_bytes = vocab * (4 if kernels.default_dtype() == "float32" else 8)
        chunk_rows = max(1, CHUNK_BYTES // row_bytes // CHUNK_STEP) * CHUNK_STEP
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    rows = inputs.numel // features
    classes = _class_indices(targets, rows, vocab)
    need_grad = is_grad_enabled() and (inputs.requires_grad or weight.requires_grad or bias.requires_grad)
    factor = 1.0 / rows                                         # d(mean loss) / d(row loss)
    x = inputs.data
    row_losses = []
    dx = None
    dw = kernels.zeros(vocab * features) if need_grad and weight.requires_grad else None
    db = kernels.zeros(vocab) if need_grad and bias.requires_grad else None
    for start in range(0, rows, chunk_rows):
        count = min(chunk_rows, rows - start)
        x_chunk = x[start * features : (start + count) * features]
        logits = kernels.gemm(
            x_chunk, weight.data, batches=1, m=count, k=features, n=vocab,
            a_step=0, b_step=0, c_step=0, trans_b=True, initial=kernels.storage(bias.data) * count,
        )
        grad, nll = kernels.softmax_xent_rows(logits, classes[start : start + count], factor, vocab)
        row_losses.extend(nll)
        if not need_grad:
            continue
        if inputs.requires_grad:
            part = kernels.gemm(grad, weight.data, batches=1, m=count, k=vocab, n=features,
                                a_step=0, b_step=0, c_step=0)
            dx = part if dx is None else dx + part
        if dw is not None:
            dw = kernels.gemm(grad, x_chunk, batches=1, m=vocab, k=count, n=features,
                              a_step=0, b_step=0, c_step=0, trans_a=True, initial=dw)
        if db is not None:
            db = kernels.add_rows(grad, db, vocab)
    loss = sum(row_losses)                                      # as softmax_cross_entropy adds them
    out = _op_result([loss / rows], (1,), (inputs, weight, bias), "linear_cross_entropy")

    def _backward() -> None:
        scale = _grad_data(out)[0]
        for parent, grad in ((inputs, dx), (weight, dw), (bias, db)):
            if parent.requires_grad and grad is not None:
                _add_grad(parent, grad if scale == 1.0 else kernels.scale(grad, scale))

    if need_grad:
        _set_backward(out, _backward)
    return out


def soft_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean cross entropy against target distributions instead of class ids:

        loss = mean over rows of  -sum_c q_c log softmax(x)_c  =  logsumexp(x) - sum_c q_c x_c
        dloss/dx = softmax(x) - q                              (per row, over rows)

    (rows of ``targets`` sum to 1). Used for knowledge distillation, where
    the targets are a teacher's softened predictions.
    """

    _require_same_shape(logits, targets)
    rows, cols = logits.shape
    x, q = logits.data, targets.data
    probabilities = kernels.softmax_rows(x, cols)
    loss = 0.0
    for row in range(rows):
        values = x[row * cols : (row + 1) * cols]
        top = max(values)
        log_total = top + math.log(sum(math.exp(v - top) for v in values))
        loss += sum(q[row * cols + c] * (log_total - values[c]) for c in range(cols))
    out = _op_result([loss / rows], (1,), (logits,), "soft_cross_entropy")

    def _backward() -> None:
        grad = kernels.add(probabilities, kernels.scale(q, -1.0))
        _add_grad(logits, kernels.scale(grad, _grad_data(out)[0] / rows))

    _set_backward(out, _backward)
    return out


def _target_nll(x, probabilities, row: int, cols: int, target: int) -> float:
    """-log p_target. When p_target underflows to 0 (a confident, wrong
    prediction), compute it from the logits instead: logsumexp(x) - x_target."""

    p = probabilities[row * cols + target]
    if p > 0.0:
        return -math.log(p)
    values = x[row * cols : (row + 1) * cols]
    top = max(values)
    return top + math.log(math.fsum(math.exp(value - top) for value in values)) - x[row * cols + target]


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
