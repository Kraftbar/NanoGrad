"""Compute kernels with two interchangeable backends: C and pure Python.

The heavy inner loops (batched matrix multiplication, im2col, pooling,
GELU, tanh, row softmax and row LayerNorm, forward and backward) live here so ``tensor.py`` can stay
about the math. Two backends compute the same thing:

- ``"c"``: ``kernels.c``, compiled with the system C compiler (``$CC`` or
  ``cc``) the first time it is needed and loaded with ``ctypes``. The build
  is cached under ``~/.cache/nanograd/``.
- ``"python"``: plain Python dot products, the readable reference.

The C backend is used when it builds; otherwise, or when the environment
variable ``NANOGRAD_PURE_PYTHON=1`` is set, everything runs in Python.
``BACKEND`` says which one is active, and ``use_backend`` switches (tests
use it to check that both agree).
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
import shutil
import subprocess
import tempfile
from array import array
from collections.abc import Callable, Sequence
from operator import mul
from pathlib import Path


SOURCE = Path(__file__).resolve().parent / "kernels.c"


# ---------------------------------------------------------------------------
# Pure-Python reference
# ---------------------------------------------------------------------------


def _python_dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(map(mul, left, right))


# math.sumprod (Python 3.12+) is a C-speed dot product.
dot: Callable[[Sequence[float], Sequence[float]], float] = getattr(math, "sumprod", _python_dot)


def _python_gemm(
    a: list[float],
    b: list[float],
    batches: int,
    m: int,
    k: int,
    n: int,
    a_step: int,
    b_step: int,
    c_step: int,
    trans_a: bool,
    trans_b: bool,
    scratch: bool = False,
) -> list[float]:
    """Reference ``gemm``: every output entry is one dot product.

    Row i of op(A) and column j of op(B) are read off as list slices:
    a row of a row-major matrix is a contiguous slice, a column is a slice
    with a stride.
    """

    size = c_step * (batches - 1) + m * n if c_step else m * n
    c = [0.0] * size
    for batch in range(batches):
        a_start = batch * a_step
        b_start = batch * b_step
        if trans_a:   # A stored as (k x m): row i of A^T is column i of A
            a_rows = [a[a_start + i : a_start + k * m : m] for i in range(m)]
        else:
            a_rows = [a[a_start + i * k : a_start + (i + 1) * k] for i in range(m)]
        if trans_b:   # B stored as (n x k): column j of B^T is row j of B
            b_columns = [b[b_start + j * k : b_start + (j + 1) * k] for j in range(n)]
        else:
            b_columns = [b[b_start + j : b_start + k * n : n] for j in range(n)]

        c_start = batch * c_step
        for i, row in enumerate(a_rows):
            offset = c_start + i * n
            for j, column in enumerate(b_columns):
                c[offset + j] += dot(row, column)
    return c


def _python_gelu(x: list[float]) -> tuple[list[float], list[float]]:
    cdf = [0.5 * (1.0 + math.erf(value / _SQRT_TWO)) for value in x]
    return [value * phi for value, phi in zip(x, cdf)], cdf


def _python_gelu_backward(x: list[float], cdf: list[float], grad: list[float]) -> list[float]:
    return [
        g * (phi + value * math.exp(-0.5 * value * value) / _SQRT_TWO_PI)
        for g, value, phi in zip(grad, x, cdf)
    ]


def _python_softmax_rows(x: list[float], width: int) -> list[float]:
    out: list[float] = []
    for start in range(0, len(x), width):
        row = x[start : start + width]
        row_max = max(row)
        exps = [math.exp(value - row_max) for value in row]
        total = sum(exps)
        out.extend([value / total for value in exps])
    return out


def _python_softmax_rows_backward(y: list[float], grad: list[float], width: int) -> list[float]:
    out: list[float] = []
    for start in range(0, len(y), width):
        probabilities = y[start : start + width]
        row_grad = grad[start : start + width]
        weighted = dot(row_grad, probabilities)
        out.extend([p * (g - weighted) for p, g in zip(probabilities, row_grad)])
    return out


def _python_layer_norm_rows(
    x: list[float], weight: list[float], bias: list[float], width: int, eps: float,
) -> tuple[list[float], list[float], list[float]]:
    y: list[float] = []
    x_hat: list[float] = []
    inv_std: list[float] = []
    for start in range(0, len(x), width):
        row = x[start : start + width]
        mean = sum(row) / width
        centered = [value - mean for value in row]
        variance = sum(value * value for value in centered) / width
        scale = 1.0 / math.sqrt(variance + eps)
        row_hat = [value * scale for value in centered]
        inv_std.append(scale)
        x_hat.extend(row_hat)
        y.extend([h * w + b for h, w, b in zip(row_hat, weight, bias)])
    return y, x_hat, inv_std


def _python_layer_norm_rows_backward(
    grad: list[float], x_hat: list[float], inv_std: list[float], weight: list[float],
    width: int, need_dx: bool,
) -> tuple[list[float] | None, list[float], list[float]]:
    dx: list[float] = []
    dweight = [0.0] * width
    dbias = [0.0] * width
    for row, scale in enumerate(inv_std):
        dy = grad[row * width : (row + 1) * width]
        h = x_hat[row * width : (row + 1) * width]
        dweight = [total + a * b for total, a, b in zip(dweight, dy, h)]
        dbias = [total + a for total, a in zip(dbias, dy)]
        if need_dx:
            g = [a * w for a, w in zip(dy, weight)]
            mean_g = sum(g) / width
            mean_g_h = dot(g, h) / width
            dx.extend([scale * (gi - mean_g - hi * mean_g_h) for gi, hi in zip(g, h)])
    return (dx if need_dx else None), dweight, dbias


def _python_gather_indexed(
    image: list[float], index: Sequence[int], batches: int, image_size: int, scratch: bool = False,
) -> list[float]:
    out: list[float] = []
    for batch in range(batches):
        start = batch * image_size
        single = image[start : start + image_size]
        out.extend([single[i] for i in index])
    return out


def _python_scatter_add_indexed(
    grad: list[float], index: Sequence[int], batches: int, image_size: int,
) -> list[float]:
    image_grad = [0.0] * (batches * image_size)
    count = len(index)
    for batch in range(batches):
        start = batch * image_size
        values = grad[batch * count : (batch + 1) * count]
        for i, value in zip(index, values):
            image_grad[start + i] += value
    return image_grad


def _python_tanh(x: list[float]) -> list[float]:
    return [math.tanh(value) for value in x]


def _python_tanh_backward(y: list[float], grad: list[float]) -> list[float]:
    return [g * (1.0 - value * value) for g, value in zip(grad, y)]


def _python_window_mean(
    image: list[float], index: Sequence[int], planes: int, plane_size: int, window_size: int,
) -> list[float]:
    out: list[float] = []
    for plane in range(planes):
        start = plane * plane_size
        cells = [image[start + i] for i in index]
        out.extend([
            sum(cells[w : w + window_size]) / window_size
            for w in range(0, len(cells), window_size)
        ])
    return out


def _python_window_mean_backward(
    grad: list[float], index: Sequence[int], planes: int, plane_size: int, window_size: int,
) -> list[float]:
    image_grad = [0.0] * (planes * plane_size)
    windows = len(index) // window_size
    for plane in range(planes):
        start = plane * plane_size
        for w in range(windows):
            share = grad[plane * windows + w] / window_size
            for i in index[w * window_size : (w + 1) * window_size]:
                image_grad[start + i] += share
    return image_grad


def _python_window_max(
    image: list[float], index: Sequence[int], planes: int, plane_size: int, window_size: int,
) -> tuple[list[float], list[int]]:
    out: list[float] = []
    argmax: list[int] = []
    for plane in range(planes):
        start = plane * plane_size
        for w in range(0, len(index), window_size):
            # max() keeps the first maximum, matching the C loop's strict ">".
            best = max(index[w : w + window_size], key=lambda i: image[start + i])
            out.append(image[start + best])
            argmax.append(start + best)
    return out, argmax


_SQRT_TWO = math.sqrt(2.0)
_SQRT_TWO_PI = math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# C backend
# ---------------------------------------------------------------------------


def _build_library() -> ctypes.CDLL | None:
    """Compile kernels.c (once per source version) and load it, or None."""

    compiler = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if compiler is None or not SOURCE.exists():
        return None

    source = SOURCE.read_bytes()
    digest = hashlib.sha256(source).hexdigest()[:16]
    cache = Path(os.environ.get("NANOGRAD_CACHE", Path.home() / ".cache" / "nanograd"))
    library_path = cache / f"kernels-{digest}.so"
    if not library_path.exists():
        try:
            cache.mkdir(parents=True, exist_ok=True)
            # Build into a temporary name, then rename: two processes
            # building at once never see a half-written library.
            with tempfile.NamedTemporaryFile(dir=cache, suffix=".so", delete=False) as handle:
                partial = Path(handle.name)
            subprocess.run(
                [compiler, "-O3", "-shared", "-fPIC", str(SOURCE), "-o", str(partial), "-lm"],
                check=True,
                capture_output=True,
            )
            partial.replace(library_path)
        except (OSError, subprocess.CalledProcessError):
            return None

    try:
        library = ctypes.CDLL(str(library_path))
    except OSError:
        return None
    pointer = ctypes.POINTER(ctypes.c_double)
    library.gemm.argtypes = [
        pointer, pointer, pointer,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_long, ctypes.c_long, ctypes.c_long,
        ctypes.c_int, ctypes.c_int,
    ]
    library.gemm.restype = None
    count = ctypes.c_long
    library.gelu_forward.argtypes = [pointer, pointer, pointer, count]
    library.gelu_backward.argtypes = [pointer, pointer, pointer, pointer, count]
    library.softmax_rows.argtypes = [pointer, pointer, count, count]
    library.softmax_rows_backward.argtypes = [pointer, pointer, pointer, count, count]
    library.layer_norm_rows.argtypes = [pointer] * 6 + [count, count, ctypes.c_double]
    library.layer_norm_rows_backward.argtypes = [pointer] * 7 + [count, count, ctypes.c_int]
    index_pointer = ctypes.POINTER(ctypes.c_long)
    for name in ("gather_indexed", "scatter_add_indexed"):
        getattr(library, name).argtypes = [pointer, index_pointer, pointer, count, count, count]
    library.tanh_forward.argtypes = [pointer, pointer, count]
    library.tanh_backward.argtypes = [pointer, pointer, pointer, count]
    for name in ("window_mean", "window_mean_backward"):
        getattr(library, name).argtypes = [pointer, index_pointer, pointer] + [count] * 4
    library.window_max.argtypes = [pointer, index_pointer, pointer, index_pointer] + [count] * 4
    return library


def _c_buffer(values: array):
    return (ctypes.c_double * len(values)).from_buffer(values)


def _doubles(values: Sequence[float]) -> array:
    """The values as a C double array; an array passes through uncopied."""

    return values if isinstance(values, array) else array("d", values)


def _result(values: array, scratch: bool) -> array | list[float]:
    """Tensors hold lists; ``scratch`` results only feed other kernels."""

    return values if scratch else values.tolist()


def _c_gemm(
    a: list[float],
    b: list[float],
    batches: int,
    m: int,
    k: int,
    n: int,
    a_step: int,
    b_step: int,
    c_step: int,
    trans_a: bool,
    trans_b: bool,
    scratch: bool = False,
) -> list[float]:
    size = c_step * (batches - 1) + m * n if c_step else m * n
    a_buffer = _doubles(a)
    b_buffer = _doubles(b)
    c_buffer = _zeros(size)
    _library.gemm(
        _c_buffer(a_buffer), _c_buffer(b_buffer), _c_buffer(c_buffer),
        batches, m, k, n,
        a_step, b_step, c_step,
        int(trans_a), int(trans_b),
    )
    return _result(c_buffer, scratch)


_ZERO = array("d", [0.0])


def _zeros(count: int) -> array:
    return _ZERO * count   # repeating a one-element array is a fast fill


def _c_gelu(x: list[float]) -> tuple[list[float], list[float]]:
    x_buffer, y, cdf = _doubles(x), _zeros(len(x)), _zeros(len(x))
    _library.gelu_forward(_c_buffer(x_buffer), _c_buffer(y), _c_buffer(cdf), len(x))
    return y.tolist(), cdf.tolist()


def _c_gelu_backward(x: list[float], cdf: list[float], grad: list[float]) -> list[float]:
    dx = _zeros(len(x))
    buffers = [_doubles(values) for values in (x, cdf, grad)]
    _library.gelu_backward(*map(_c_buffer, buffers), _c_buffer(dx), len(x))
    return dx.tolist()


def _c_softmax_rows(x: list[float], width: int) -> list[float]:
    x_buffer, y = _doubles(x), _zeros(len(x))
    _library.softmax_rows(_c_buffer(x_buffer), _c_buffer(y), len(x) // width, width)
    return y.tolist()


def _c_softmax_rows_backward(y: list[float], grad: list[float], width: int) -> list[float]:
    y_buffer, grad_buffer, dx = _doubles(y), _doubles(grad), _zeros(len(y))
    _library.softmax_rows_backward(
        _c_buffer(y_buffer), _c_buffer(grad_buffer), _c_buffer(dx), len(y) // width, width,
    )
    return dx.tolist()


def _c_layer_norm_rows(
    x: list[float], weight: list[float], bias: list[float], width: int, eps: float,
) -> tuple[list[float], list[float], list[float]]:
    rows = len(x) // width
    inputs = [_doubles(values) for values in (x, weight, bias)]
    y, x_hat, inv_std = _zeros(len(x)), _zeros(len(x)), _zeros(rows)
    _library.layer_norm_rows(
        *map(_c_buffer, inputs), _c_buffer(y), _c_buffer(x_hat), _c_buffer(inv_std), rows, width, eps,
    )
    return y.tolist(), x_hat.tolist(), inv_std.tolist()


def _c_layer_norm_rows_backward(
    grad: list[float], x_hat: list[float], inv_std: list[float], weight: list[float],
    width: int, need_dx: bool,
) -> tuple[list[float] | None, list[float], list[float]]:
    inputs = [_doubles(values) for values in (grad, x_hat, inv_std, weight)]
    dx = _zeros(len(grad) if need_dx else 1)
    dweight, dbias = _zeros(width), _zeros(width)
    _library.layer_norm_rows_backward(
        *map(_c_buffer, inputs), _c_buffer(dx), _c_buffer(dweight), _c_buffer(dbias),
        len(inv_std), width, int(need_dx),
    )
    return (dx.tolist() if need_dx else None), dweight.tolist(), dbias.tolist()


def _c_index_buffer(index: Sequence[int]):
    values = array("l", index)
    return values, (ctypes.c_long * len(values)).from_buffer(values)


def _c_gather_indexed(
    image: list[float], index: Sequence[int], batches: int, image_size: int, scratch: bool = False,
) -> list[float]:
    image_buffer = _doubles(image)
    keep, index_buffer = _c_index_buffer(index)
    out = _zeros(batches * len(index))
    _library.gather_indexed(
        _c_buffer(image_buffer), index_buffer, _c_buffer(out), batches, image_size, len(index),
    )
    return _result(out, scratch)


def _c_scatter_add_indexed(
    grad: list[float], index: Sequence[int], batches: int, image_size: int,
) -> list[float]:
    grad_buffer = _doubles(grad)
    keep, index_buffer = _c_index_buffer(index)
    image_grad = _zeros(batches * image_size)
    _library.scatter_add_indexed(
        _c_buffer(grad_buffer), index_buffer, _c_buffer(image_grad), batches, image_size, len(index),
    )
    return image_grad.tolist()


def _c_tanh(x: list[float]) -> list[float]:
    x_buffer, y = _doubles(x), _zeros(len(x))
    _library.tanh_forward(_c_buffer(x_buffer), _c_buffer(y), len(x))
    return y.tolist()


def _c_tanh_backward(y: list[float], grad: list[float]) -> list[float]:
    y_buffer, grad_buffer, dx = _doubles(y), _doubles(grad), _zeros(len(y))
    _library.tanh_backward(_c_buffer(y_buffer), _c_buffer(grad_buffer), _c_buffer(dx), len(y))
    return dx.tolist()


def _c_window_mean(
    image: list[float], index: Sequence[int], planes: int, plane_size: int, window_size: int,
) -> list[float]:
    image_buffer = _doubles(image)
    keep, index_buffer = _c_index_buffer(index)
    windows = len(index) // window_size
    out = _zeros(planes * windows)
    _library.window_mean(
        _c_buffer(image_buffer), index_buffer, _c_buffer(out), planes, plane_size, windows, window_size,
    )
    return out.tolist()


def _c_window_mean_backward(
    grad: list[float], index: Sequence[int], planes: int, plane_size: int, window_size: int,
) -> list[float]:
    grad_buffer = _doubles(grad)
    keep, index_buffer = _c_index_buffer(index)
    image_grad = _zeros(planes * plane_size)
    _library.window_mean_backward(
        _c_buffer(grad_buffer), index_buffer, _c_buffer(image_grad),
        planes, plane_size, len(index) // window_size, window_size,
    )
    return image_grad.tolist()


def _c_window_max(
    image: list[float], index: Sequence[int], planes: int, plane_size: int, window_size: int,
) -> tuple[list[float], list[int]]:
    image_buffer = _doubles(image)
    keep, index_buffer = _c_index_buffer(index)
    windows = len(index) // window_size
    out = _zeros(planes * windows)
    argmax, argmax_buffer = _c_index_buffer([0] * (planes * windows))
    _library.window_max(
        _c_buffer(image_buffer), index_buffer, _c_buffer(out), argmax_buffer,
        planes, plane_size, windows, window_size,
    )
    return out.tolist(), argmax.tolist()


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


_library = None if os.environ.get("NANOGRAD_PURE_PYTHON") == "1" else _build_library()
BACKEND = "c" if _library is not None else "python"


def use_backend(name: str) -> None:
    """Switch to ``"c"`` or ``"python"`` (``"c"`` only if it was built)."""

    global BACKEND, _library
    if name == "python":
        BACKEND = "python"
        return
    if name != "c":
        raise ValueError("backend must be 'c' or 'python'")
    if _library is None:
        _library = _build_library()
    if _library is None:
        raise RuntimeError("the C kernels could not be built (is a C compiler installed?)")
    BACKEND = "c"


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
    scratch: bool = False,
) -> list[float]:
    """Batched ``C = op(A) @ op(B)``; see ``kernels.c`` for the conventions.

    A ``c_step`` of 0 sums all batch products into one (m x n) result.
    ``scratch=True`` is for results that only feed other kernels: the C
    backend then keeps them as a C array instead of copying to a list.
    """

    kernel = _c_gemm if BACKEND == "c" else _python_gemm
    return kernel(a, b, batches, m, k, n, a_step, b_step, c_step, trans_a, trans_b, scratch)


def gelu(x: list[float]) -> tuple[list[float], list[float]]:
    """GELU of every value, plus Phi(x) (the normal CDF) for the backward pass."""

    return (_c_gelu if BACKEND == "c" else _python_gelu)(x)


def gelu_backward(x: list[float], cdf: list[float], grad: list[float]) -> list[float]:
    return (_c_gelu_backward if BACKEND == "c" else _python_gelu_backward)(x, cdf, grad)


def softmax_rows(x: list[float], width: int) -> list[float]:
    """Softmax of every contiguous row of ``width`` values."""

    return (_c_softmax_rows if BACKEND == "c" else _python_softmax_rows)(x, width)


def softmax_rows_backward(y: list[float], grad: list[float], width: int) -> list[float]:
    kernel = _c_softmax_rows_backward if BACKEND == "c" else _python_softmax_rows_backward
    return kernel(y, grad, width)


def layer_norm_rows(
    x: list[float], weight: list[float], bias: list[float], width: int, eps: float,
) -> tuple[list[float], list[float], list[float]]:
    """LayerNorm of every row: returns (y, x_hat, inv_std per row)."""

    kernel = _c_layer_norm_rows if BACKEND == "c" else _python_layer_norm_rows
    return kernel(x, weight, bias, width, eps)


def layer_norm_rows_backward(
    grad: list[float], x_hat: list[float], inv_std: list[float], weight: list[float],
    width: int, need_dx: bool,
) -> tuple[list[float] | None, list[float], list[float]]:
    """Returns (dx or None, dweight, dbias), dweight and dbias summed over rows."""

    kernel = _c_layer_norm_rows_backward if BACKEND == "c" else _python_layer_norm_rows_backward
    return kernel(grad, x_hat, inv_std, weight, width, need_dx)


def gather_indexed(
    image: list[float], index: Sequence[int], *, batches: int, image_size: int, scratch: bool = False,
) -> list[float]:
    """For every image in the batch, read the pixels at ``index`` (im2col).

    ``scratch`` works as in ``gemm``.
    """

    kernel = _c_gather_indexed if BACKEND == "c" else _python_gather_indexed
    return kernel(image, index, batches, image_size, scratch)


def scatter_add_indexed(grad: list[float], index: Sequence[int], *, batches: int, image_size: int) -> list[float]:
    """Inverse of ``gather_indexed`` for gradients: add values back (col2im)."""

    kernel = _c_scatter_add_indexed if BACKEND == "c" else _python_scatter_add_indexed
    return kernel(grad, index, batches, image_size)


def tanh(x: list[float]) -> list[float]:
    return (_c_tanh if BACKEND == "c" else _python_tanh)(x)


def tanh_backward(y: list[float], grad: list[float]) -> list[float]:
    """Gradient of tanh from its output: ``grad * (1 - y^2)``."""

    return (_c_tanh_backward if BACKEND == "c" else _python_tanh_backward)(y, grad)


def window_mean(
    image: list[float], index: Sequence[int], *, planes: int, plane_size: int, window_size: int,
) -> list[float]:
    """Average pooling: the mean of every window (``index`` lists them back to back)."""

    kernel = _c_window_mean if BACKEND == "c" else _python_window_mean
    return kernel(image, index, planes, plane_size, window_size)


def window_mean_backward(
    grad: list[float], index: Sequence[int], *, planes: int, plane_size: int, window_size: int,
) -> list[float]:
    """Every pixel of a window gets an equal share of that window's gradient."""

    kernel = _c_window_mean_backward if BACKEND == "c" else _python_window_mean_backward
    return kernel(grad, index, planes, plane_size, window_size)


def window_max(
    image: list[float], index: Sequence[int], *, planes: int, plane_size: int, window_size: int,
) -> tuple[list[float], list[int]]:
    """Max pooling: the max of every window, and where it came from in ``image``."""

    kernel = _c_window_max if BACKEND == "c" else _python_window_max
    return kernel(image, index, planes, plane_size, window_size)
