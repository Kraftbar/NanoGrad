"""Compute kernels: the inner loops, with two interchangeable backends.

``tensor.py`` says what each op computes; the loops that do the heavy work
(matrix products, im2col, pooling, GELU, tanh, row softmax, row LayerNorm,
elementwise arithmetic and the Adam update, forward and backward) live in
the kernels. Each one exists twice:

- ``kernels_reference.py``: pure Python, written to be read.
- ``kernels.c``: the same loops in C, built and called by ``kernels_c.py``.

This module picks one. The C backend is used when it builds (it needs a C
compiler); otherwise, or when ``NANOGRAD_PURE_PYTHON=1`` is set, everything
runs in Python. The C kernels are built for this exact CPU
(``-march=native``); ``NANOGRAD_CFLAGS=""`` asks for a portable build. ``BACKEND`` says which backend is active, and
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

    The C kernels read and write C arrays (``array("d")`` for float64, or
    ``array("f")`` after ``set_default_dtype("float32")``), so with the C
    backend tensors keep their values in one: data then flows from kernel
    to kernel without being copied into Python lists and back.
    The Python backend keeps plain lists. Both are flat sequences of floats
    that index, slice and iterate the same way.
    """

    if BACKEND == "c":
        code = kernels_c.typecode
        return values if type(values) is array and values.typecode == code else array(code, values)
    return values if type(values) is list else list(values)


def set_default_dtype(name: str) -> None:
    """Store new tensors as "float64" (the default) or "float32" C arrays.

    float32 halves memory traffic and doubles how many numbers each vector
    instruction handles, so the C kernels run about twice as fast; results
    are then accurate to about 7 digits instead of 16. Like
    ``torch.set_default_dtype``, it affects tensors created afterwards. The
    Python backend always computes in float64.
    """

    codes = {"float64": "d", "float32": "f"}
    if name not in codes:
        raise ValueError("dtype must be 'float64' or 'float32'")
    if BACKEND != "c":
        return                       # the Python backend always uses float64
    if not kernels_c.set_precision(codes[name]) and name == "float32":
        raise RuntimeError("the float32 C kernels could not be built")


def default_dtype() -> str:
    return "float32" if BACKEND == "c" and kernels_c.typecode == "f" else "float64"


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
    initial: list[float] | None = None,
) -> list[float]:
    """Batched ``C = op(A) @ op(B)``; see ``kernels.c`` for the conventions.

    A ``c_step`` of 0 sums all batch products into one (m x n) result.
    With ``initial`` the products are added to it instead of to zeros
    (``C = initial + op(A) @ op(B)``, which is how a linear layer adds its
    bias without a separate pass). ``initial`` is consumed: a C array of the
    current precision becomes the result itself, with no copy.
    """

    kernel = kernels_c.gemm if BACKEND == "c" else reference.gemm
    return kernel(a, b, batches, m, k, n, a_step, b_step, c_step, trans_a, trans_b, initial)


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


def batch_norm_channels(
    x: list[float], weight: list[float], bias: list[float], *,
    batch: int, channels: int, plane: int, eps: float,
):
    """Training-mode batch norm of (batch, channels, plane) data: returns
    (y, x_hat, inv_std, mean, variance), the last three per channel."""

    kernel = kernels_c.batch_norm_channels if BACKEND == "c" else reference.batch_norm_channels
    return kernel(x, weight, bias, batch, channels, plane, eps)


def batch_norm_channels_backward(
    grad: list[float], x_hat: list[float], inv_std: list[float], weight: list[float], *,
    batch: int, channels: int, plane: int,
):
    """Returns (dx, dweight, dbias)."""

    kernel = kernels_c.batch_norm_channels_backward if BACKEND == "c" else reference.batch_norm_channels_backward
    return kernel(grad, x_hat, inv_std, weight, batch, channels, plane)


def conv_forward(
    image: list[float], index: Sequence[int], kernel: list[float], *,
    batches: int, image_size: int, filters: int, patch_size: int, positions: int,
) -> list[float]:
    """Convolution as kernel @ patches^T, reading patches through ``index``
    (see conv2d_valid) without building the patch matrix."""

    kernel_fn = kernels_c.conv_forward if BACKEND == "c" else reference.conv_forward
    return kernel_fn(image, index, kernel, batches, image_size, filters, patch_size, positions)


def conv_kernel_grad(
    image: list[float], index: Sequence[int], grad: list[float], *,
    batches: int, image_size: int, filters: int, patch_size: int, positions: int,
) -> list[float]:
    """Gradient of conv_forward's kernel, summed over the batch, read through
    ``index`` without building the patch matrix."""

    kernel_fn = kernels_c.conv_kernel_grad if BACKEND == "c" else reference.conv_kernel_grad
    return kernel_fn(image, index, grad, batches, image_size, filters, patch_size, positions)


def conv_direct(
    image: list[float], kernel: list[float], *, batches: int, channels: int, height: int, width: int,
    filters: int, kernel_rows: int, kernel_cols: int,
) -> list[float]:
    """Stride-1 valid convolution computed directly, without patches (fast
    for few filters): out[b, f, y, x] = sum of kernel[f, c, i, j] * image[b, c, y+i, x+j]."""

    kernel_fn = kernels_c.conv_direct if BACKEND == "c" else reference.conv_direct
    return kernel_fn(image, kernel, batches, channels, height, width, filters, kernel_rows, kernel_cols)


def conv_input_grad(
    grad: list[float], kernel: list[float], index: Sequence[int], *,
    batches: int, image_size: int, filters: int, patch_size: int, positions: int,
) -> list[float]:
    """Gradient of conv_forward's image: patch gradients scattered back onto
    their pixels (col2im), without a patch-gradient matrix."""

    kernel_fn = kernels_c.conv_input_grad if BACKEND == "c" else reference.conv_input_grad
    return kernel_fn(grad, kernel, index, batches, image_size, filters, patch_size, positions)


def gather_indexed(image: list[float], index: Sequence[int], *, batches: int, image_size: int) -> list[float]:
    """For every image in the batch, read the pixels at ``index`` (im2col)."""

    kernel = kernels_c.gather_indexed if BACKEND == "c" else reference.gather_indexed
    return kernel(image, index, batches, image_size)


def depthwise_forward(image, index, kernel, *, batches: int, channels: int, channel_size: int,
                      q_size: int) -> list[float]:
    """Each channel convolved with its own kernel (see kernels.c)."""

    kernel_fn = kernels_c.depthwise_forward if BACKEND == "c" else reference.depthwise_forward
    return kernel_fn(image, index, kernel, batches, channels, channel_size, q_size)


def depthwise_backward(image, index, kernel, grad, *, batches: int, channels: int, channel_size: int,
                       q_size: int):
    """Gradients (for the kernel, for the image) of ``depthwise_forward``."""

    kernel_fn = kernels_c.depthwise_backward if BACKEND == "c" else reference.depthwise_backward
    return kernel_fn(image, index, kernel, grad, batches, channels, channel_size, q_size)


def csr_matmul(row_start, cols, values, dense, *, rows: int, n: int) -> list[float]:
    """Sparse (rows x inner, stored by rows) @ dense (inner x n); see kernels.c."""

    return (kernels_c.csr_matmul if BACKEND == "c" else reference.csr_matmul)(row_start, cols, values, dense, rows, n)


def gather_rows(table: list[float], ids: Sequence[int], *, width: int) -> list[float]:
    """Rows ``ids`` of a (rows x width) table, one after another (an embedding lookup)."""

    return (kernels_c.gather_rows if BACKEND == "c" else reference.gather_rows)(table, ids, width)


def scatter_add_rows(grad: list[float], ids: Sequence[int], *, width: int, rows: int, into=None) -> list[float]:
    """Inverse of ``gather_rows`` for gradients: a zero table with each
    gradient row added into row ``ids[i]`` (repeated ids add up).

    With ``into`` (a table in the backend's storage form, owned by the
    caller) the rows are added onto it in place instead, and it is returned.
    """

    kernel = kernels_c.scatter_add_rows if BACKEND == "c" else reference.scatter_add_rows
    return kernel(grad, ids, width, rows, into)


def scatter_add_indexed(
    grad: list[float], index: Sequence[int], *, batches: int, image_size: int, into=None,
) -> list[float]:
    """Inverse of ``gather_indexed`` for gradients: add values back (col2im).

    With ``into`` (a buffer from ``zeros``, owned by the caller) the values
    are added to it in place instead of to a new zero buffer.
    """

    kernel = kernels_c.scatter_add_indexed if BACKEND == "c" else reference.scatter_add_indexed
    return kernel(grad, index, batches, image_size, into)


def add(x: list[float], y: list[float]) -> list[float]:
    """``x + y``, elementwise."""

    return (kernels_c.add if BACKEND == "c" else reference.add)(x, y)


def multiply(x: list[float], y: list[float]) -> list[float]:
    """``x * y``, elementwise."""

    return (kernels_c.multiply if BACKEND == "c" else reference.multiply)(x, y)


def scale(x: list[float], factor: float) -> list[float]:
    """Every value times ``factor``."""

    return (kernels_c.scale if BACKEND == "c" else reference.scale)(x, factor)


def reciprocal(x: list[float]) -> list[float]:
    """1 / x, elementwise."""

    return (kernels_c.reciprocal if BACKEND == "c" else reference.reciprocal)(x)


def exp(x: list[float]) -> list[float]:
    """e^x, elementwise."""

    return (kernels_c.exp if BACKEND == "c" else reference.exp)(x)


def log(x: list[float]) -> list[float]:
    """ln x, elementwise (x > 0)."""

    return (kernels_c.log if BACKEND == "c" else reference.log)(x)


def sqrt(x: list[float]) -> list[float]:
    """The square root, elementwise (x >= 0)."""

    return (kernels_c.sqrt if BACKEND == "c" else reference.sqrt)(x)


def dropout_mask(count: int, p: float, seed: int) -> list[float]:
    """``count`` values, each 0 with probability ``p`` and 1 / (1 - p) otherwise,
    drawn from a 64-bit ``seed`` (the same mask on either backend)."""

    return (kernels_c.dropout_mask if BACKEND == "c" else reference.dropout_mask)(count, p, seed)


def add_scaled(x: list[float], y: list[float], factor: float) -> list[float]:
    """``x + factor * y``, elementwise (one pass, no temporary)."""

    return (kernels_c.add_scaled if BACKEND == "c" else reference.add_scaled)(x, y, factor)


def projected_sign_step(delta: list[float], grad: list[float], x: list[float], alpha: float,
                        eps: float) -> list[float]:
    """One step of a sign-gradient attack on images ``x`` in [0, 1]:
    ``delta + alpha * sign(grad)``, clipped to [-eps, eps], then to [-x, 1 - x]
    so that ``x + delta`` stays a valid image."""

    return (kernels_c.projected_sign_step if BACKEND == "c" else reference.projected_sign_step)(
        delta, grad, x, alpha, eps)


def sum_of_squares(x: list[float]) -> float:
    """The sum of every value squared (a squared norm)."""

    return (kernels_c.sum_of_squares if BACKEND == "c" else reference.sum_of_squares)(x)


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


def sigmoid(x: list[float]) -> list[float]:
    return (kernels_c.sigmoid if BACKEND == "c" else reference.sigmoid)(x)


def sigmoid_backward(y: list[float], grad: list[float]) -> list[float]:
    """Gradient of sigmoid from its output: ``grad * y * (1 - y)``."""

    return (kernels_c.sigmoid_backward if BACKEND == "c" else reference.sigmoid_backward)(y, grad)


def lstm_memory(gates, c, *, batch: int, hidden: int) -> list[float]:
    """LSTM memory update c' = sigmoid(f) * c + sigmoid(i) * tanh(g); gates (batch, 4H) as i, f, g, o."""

    if BACKEND == "c":
        return kernels_c.lstm_step("lstm_memory", gates, c, batch, hidden)
    return reference.lstm_memory(gates, c, batch, hidden)


def lstm_memory_backward(gates, c, grad, *, batch: int, hidden: int):
    """Gradients (for the gates, for c) of ``lstm_memory``."""

    if BACKEND == "c":
        return kernels_c.lstm_step_backward("lstm_memory_backward", gates, c, grad, batch, hidden)
    return reference.lstm_memory_backward(gates, c, grad, batch, hidden)


def lstm_output(gates, c_new, *, batch: int, hidden: int) -> list[float]:
    """LSTM output h = sigmoid(o) * tanh(c')."""

    if BACKEND == "c":
        return kernels_c.lstm_step("lstm_output", gates, c_new, batch, hidden)
    return reference.lstm_output(gates, c_new, batch, hidden)


def lstm_output_backward(gates, c_new, grad, *, batch: int, hidden: int):
    """Gradients (for the gates, for c') of ``lstm_output``."""

    if BACKEND == "c":
        return kernels_c.lstm_step_backward("lstm_output_backward", gates, c_new, grad, batch, hidden)
    return reference.lstm_output_backward(gates, c_new, grad, batch, hidden)


def bce_with_logits_sum(x: list[float], t: list[float]) -> float:
    """Summed binary cross entropy of logits x against targets t (see losses.py)."""

    return (kernels_c.bce_with_logits_sum if BACKEND == "c" else reference.bce_with_logits_sum)(x, t)


def bce_with_logits_backward(x: list[float], t: list[float], g: float) -> list[float]:
    """Its gradient for the logits, times the output gradient g: g * (sigmoid(x) - t)."""

    return (kernels_c.bce_with_logits_backward if BACKEND == "c" else reference.bce_with_logits_backward)(x, t, g)


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


# NANOGRAD_DTYPE=float32 starts in float32 (see set_default_dtype).
if BACKEND == "c" and os.environ.get("NANOGRAD_DTYPE"):
    set_default_dtype(os.environ["NANOGRAD_DTYPE"])
