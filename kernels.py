"""Compute kernels: the inner loops, with two interchangeable backends.

``tensor.py`` says what each op computes; the loops that do the heavy work
(matrix products, im2col, pooling, GELU, tanh, row softmax, row LayerNorm,
elementwise arithmetic and the Adam update, forward and backward) live in
the kernels. Each one exists twice:

- ``kernels_reference.py``: pure Python, written to be read.
- ``kernels.c``: the same loops in C, built and called by ``kernels_c.py``.

This module picks one. The C backend is used when it builds (it needs a C
compiler); otherwise, or when ``NANOGRAD_PURE_PYTHON=1`` is set, everything
runs in Python. ``NANOGRAD_CFLAGS="-march=native"`` builds the C kernels
for this exact CPU. ``BACKEND`` says which backend is active, and
``use_backend`` switches (``test_kernels.py`` uses it to check that both
agree).
"""

from __future__ import annotations

import os
from array import array
from collections.abc import Sequence

import kernels_c
import kernels_reference as reference


# A C-speed dot product (math.sumprod on Python 3.12+).
dot = reference.dot


# ---------------------------------------------------------------------------
# Backend selection and tensor storage
# ---------------------------------------------------------------------------




_pure_python = os.environ.get("NANOGRAD_PURE_PYTHON") == "1"
BACKEND = "c" if not _pure_python and kernels_c.load() else "python"


def use_backend(name: str) -> None:
    """Switch to ``"c"`` or ``"python"`` (``"c"`` only if it was built)."""

    global BACKEND
    if name == "python":
        BACKEND = "python"
        return
    if name != "c":
        raise ValueError("backend must be 'c' or 'python'")
    if not kernels_c.load():
        raise RuntimeError("the C kernels could not be built (is a C compiler installed?)")
    BACKEND = "c"


def storage(values: Sequence[float]) -> array | list[float]:
    """Tensor values in the active backend's own form.

    The C kernels read and write C arrays of doubles (``array("d")``), so
    with the C backend tensors keep their values in one: data then flows
    from kernel to kernel without being copied into Python lists and back.
    The Python backend keeps plain lists. Both are flat sequences of floats
    that index, slice and iterate the same way.
    """

    if BACKEND == "c":
        return values if type(values) is array and values.typecode == "d" else array("d", values)
    return values if type(values) is list else list(values)


def index_table(indices: Sequence[int]) -> array | tuple[int, ...]:
    """A fixed table of flat offsets (im2col patches, pooling windows,
    permutations) in the form the active backend reads fastest: C longs for
    the C kernels, a tuple for Python. Callers build each table once and
    cache it."""

    return array("l", indices) if BACKEND == "c" else tuple(indices)


def zeros(count: int) -> array | list[float]:
    return kernels_c.zeros(count) if BACKEND == "c" else [0.0] * count


# ---------------------------------------------------------------------------
# The kernels
# ---------------------------------------------------------------------------


def gemm(
    a: list[float],
    b: list[float],
    *,
    batches: int,
    m: int,
    k: int,
    n: int,
    a_step: int,
    b_step: int,
    c_step: int,
    trans_a: bool = False,
    trans_b: bool = False,
) -> list[float]:
    """Batched ``C = op(A) @ op(B)``; see ``kernels.c`` for the conventions.

    A ``c_step`` of 0 sums all batch products into one (m x n) result.
    """

    kernel = kernels_c.gemm if BACKEND == "c" else reference.gemm
    return kernel(a, b, batches, m, k, n, a_step, b_step, c_step, trans_a, trans_b)


def gelu(x: list[float]) -> tuple[list[float], list[float]]:
    """GELU of every value, plus Phi(x) (the normal CDF) for the backward pass."""

    return (kernels_c.gelu if BACKEND == "c" else reference.gelu)(x)


def gelu_backward(x: list[float], cdf: list[float], grad: list[float]) -> list[float]:
    return (kernels_c.gelu_backward if BACKEND == "c" else reference.gelu_backward)(x, cdf, grad)


def softmax_rows(x: list[float], width: int) -> list[float]:
    """Softmax of every contiguous row of ``width`` values."""

    return (kernels_c.softmax_rows if BACKEND == "c" else reference.softmax_rows)(x, width)


def softmax_rows_backward(y: list[float], grad: list[float], width: int) -> list[float]:
    kernel = kernels_c.softmax_rows_backward if BACKEND == "c" else reference.softmax_rows_backward
    return kernel(y, grad, width)


def layer_norm_rows(
    x: list[float], weight: list[float], bias: list[float], width: int, eps: float,
) -> tuple[list[float], list[float], list[float]]:
    """LayerNorm of every row: returns (y, x_hat, inv_std per row)."""

    kernel = kernels_c.layer_norm_rows if BACKEND == "c" else reference.layer_norm_rows
    return kernel(x, weight, bias, width, eps)


def layer_norm_rows_backward(
    grad: list[float], x_hat: list[float], inv_std: list[float], weight: list[float],
    width: int, need_dx: bool,
) -> tuple[list[float] | None, list[float], list[float]]:
    """Returns (dx or None, dweight, dbias), dweight and dbias summed over rows."""

    kernel = kernels_c.layer_norm_rows_backward if BACKEND == "c" else reference.layer_norm_rows_backward
    return kernel(grad, x_hat, inv_std, weight, width, need_dx)


def rms_norm_rows(x: list[float], weight: list[float], width: int, eps: float) -> tuple[list[float], list[float]]:
    """RMSNorm of every row: returns (y, 1 / rms per row)."""

    return (kernels_c.rms_norm_rows if BACKEND == "c" else reference.rms_norm_rows)(x, weight, width, eps)


def rms_norm_rows_backward(
    grad: list[float], x: list[float], inv_rms: list[float], weight: list[float], width: int,
) -> tuple[list[float], list[float]]:
    """Returns (dx, dweight), dweight summed over rows."""

    kernel = kernels_c.rms_norm_rows_backward if BACKEND == "c" else reference.rms_norm_rows_backward
    return kernel(grad, x, inv_rms, weight, width)


def gather_indexed(image: list[float], index: Sequence[int], *, batches: int, image_size: int) -> list[float]:
    """For every image in the batch, read the pixels at ``index`` (im2col)."""

    kernel = kernels_c.gather_indexed if BACKEND == "c" else reference.gather_indexed
    return kernel(image, index, batches, image_size)


def scatter_add_indexed(grad: list[float], index: Sequence[int], *, batches: int, image_size: int) -> list[float]:
    """Inverse of ``gather_indexed`` for gradients: add values back (col2im)."""

    kernel = kernels_c.scatter_add_indexed if BACKEND == "c" else reference.scatter_add_indexed
    return kernel(grad, index, batches, image_size)


def add(x: list[float], y: list[float]) -> list[float]:
    """``x + y``, elementwise."""

    return (kernels_c.add if BACKEND == "c" else reference.add)(x, y)


def multiply(x: list[float], y: list[float]) -> list[float]:
    """``x * y``, elementwise."""

    return (kernels_c.multiply if BACKEND == "c" else reference.multiply)(x, y)


def scale(x: list[float], factor: float) -> list[float]:
    """Every value times ``factor``."""

    return (kernels_c.scale if BACKEND == "c" else reference.scale)(x, factor)


def adam_update(
    w: list[float], grad: list[float], m: list[float], v: list[float], *,
    grad_scale: float, beta1: float, beta2: float, decay: float,
    step_size: float, sqrt_correction: float, eps: float,
) -> tuple[list[float], list[float], list[float]]:
    """One Adam step for one parameter: returns (new weights, new m, new v)."""

    kernel = kernels_c.adam_update if BACKEND == "c" else reference.adam_update
    return kernel(w, grad, m, v, grad_scale, beta1, beta2, decay, step_size, sqrt_correction, eps)


def sum_repeated(x: list[float], *, outer: int, size: int, inner: int) -> list[float]:
    """Undo a block broadcast for gradients: ``x`` is ``outer`` copies of each
    of ``size`` values repeated ``inner`` times; return the ``size`` sums."""

    kernel = kernels_c.sum_repeated if BACKEND == "c" else reference.sum_repeated
    return kernel(x, outer, size, inner)


def relu(x: list[float]) -> list[float]:
    return (kernels_c.relu if BACKEND == "c" else reference.relu)(x)


def relu_backward(x: list[float], grad: list[float]) -> list[float]:
    """Gradient of ReLU: ``grad`` where the input was positive, else 0."""

    return (kernels_c.relu_backward if BACKEND == "c" else reference.relu_backward)(x, grad)


def tanh(x: list[float]) -> list[float]:
    return (kernels_c.tanh if BACKEND == "c" else reference.tanh)(x)


def tanh_backward(y: list[float], grad: list[float]) -> list[float]:
    """Gradient of tanh from its output: ``grad * (1 - y^2)``."""

    return (kernels_c.tanh_backward if BACKEND == "c" else reference.tanh_backward)(y, grad)


def window_mean(
    image: list[float], index: Sequence[int], *, planes: int, plane_size: int, window_size: int,
) -> list[float]:
    """Average pooling: the mean of every window (``index`` lists them back to back)."""

    kernel = kernels_c.window_mean if BACKEND == "c" else reference.window_mean
    return kernel(image, index, planes, plane_size, window_size)


def window_mean_backward(
    grad: list[float], index: Sequence[int], *, planes: int, plane_size: int, window_size: int,
) -> list[float]:
    """Every pixel of a window gets an equal share of that window's gradient."""

    kernel = kernels_c.window_mean_backward if BACKEND == "c" else reference.window_mean_backward
    return kernel(grad, index, planes, plane_size, window_size)


def window_max(
    image: list[float], index: Sequence[int], *, planes: int, plane_size: int, window_size: int,
) -> tuple[list[float], list[int]]:
    """Max pooling: the max of every window, and where it came from in ``image``."""

    kernel = kernels_c.window_max if BACKEND == "c" else reference.window_max
    return kernel(image, index, planes, plane_size, window_size)
