"""Finite-difference gradient checks for tensor ops.

Autograd gives an exact gradient from hand-written backward rules. A finite
difference estimates the same gradient straight from the definition:

    df/dx_i  ~=  (f(x + h e_i) - f(x - h e_i)) / (2 h)

If the two agree for every input value, the backward rule is almost certainly
right. This is slow (two forward passes per value), so use it on tiny inputs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from tensor import Tensor


def numerical_gradient(
    function: Callable[[], Tensor],
    tensor: Tensor,
    *,
    step: float = 1e-6,
) -> list[float]:
    """Central-difference gradient of a scalar ``function()`` w.r.t. tensor."""

    gradient = []
    for index, original in enumerate(tensor.data):
        tensor.data[index] = original + step
        plus = function().data[0]
        tensor.data[index] = original - step
        minus = function().data[0]
        tensor.data[index] = original
        gradient.append((plus - minus) / (2 * step))
    return gradient


def max_gradient_error(
    function: Callable[[], Tensor],
    tensors: Sequence[Tensor],
    *,
    step: float = 1e-6,
) -> float:
    """Largest |autograd - numerical| over every value of every tensor.

    ``function`` must rebuild its graph from the tensors each call and return
    a single-value tensor (a loss).
    """

    loss = function()
    loss.backward()
    analytic = [list(tensor.grad or [0.0] * tensor.numel) for tensor in tensors]

    worst = 0.0
    for tensor, expected in zip(tensors, analytic):
        estimate = numerical_gradient(function, tensor, step=step)
        for exact, approx in zip(expected, estimate):
            worst = max(worst, abs(exact - approx))
    return worst
