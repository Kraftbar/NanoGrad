"""The pure-Python kernels: the readable version of every loop in kernels.c.

Each function here computes exactly what its C twin in ``kernels.c``
computes, written for reading rather than speed. ``kernels.py`` runs these
when the C kernels are not available (or ``NANOGRAD_PURE_PYTHON=1``), and
``test_kernels.py`` checks that both agree.

Values come in as flat row-major sequences of floats and go out as lists.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from operator import mul


def python_dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(map(mul, left, right))


# math.sumprod (Python 3.12+) is a C-speed dot product.
dot: Callable[[Sequence[float], Sequence[float]], float] = getattr(math, "sumprod", python_dot)


def gemm(
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


_SQRT_TWO = math.sqrt(2.0)
_SQRT_TWO_PI = math.sqrt(2.0 * math.pi)


def gelu(x: list[float]) -> tuple[list[float], list[float]]:
    cdf = [0.5 * (1.0 + math.erf(value / _SQRT_TWO)) for value in x]
    return [value * phi for value, phi in zip(x, cdf)], cdf


def gelu_backward(x: list[float], cdf: list[float], grad: list[float]) -> list[float]:
    return [
        g * (phi + value * math.exp(-0.5 * value * value) / _SQRT_TWO_PI)
        for g, value, phi in zip(grad, x, cdf)
    ]


def softmax_rows(x: list[float], width: int) -> list[float]:
    out: list[float] = []
    for start in range(0, len(x), width):
        row = x[start : start + width]
        row_max = max(row)
        exps = [math.exp(value - row_max) for value in row]
        total = sum(exps)
        out.extend([value / total for value in exps])
    return out


def softmax_rows_backward(y: list[float], grad: list[float], width: int) -> list[float]:
    out: list[float] = []
    for start in range(0, len(y), width):
        probabilities = y[start : start + width]
        row_grad = grad[start : start + width]
        weighted = dot(row_grad, probabilities)
        out.extend([p * (g - weighted) for p, g in zip(probabilities, row_grad)])
    return out


def layer_norm_rows(
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


def layer_norm_rows_backward(
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


def gather_indexed(
    image: list[float], index: Sequence[int], batches: int, image_size: int,
) -> list[float]:
    out: list[float] = []
    for batch in range(batches):
        start = batch * image_size
        single = image[start : start + image_size]
        out.extend([single[i] for i in index])
    return out


def scatter_add_indexed(
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


def add(x: list[float], y: list[float]) -> list[float]:
    return [a + b for a, b in zip(x, y)]


def multiply(x: list[float], y: list[float]) -> list[float]:
    return [a * b for a, b in zip(x, y)]


def scale(x: list[float], factor: float) -> list[float]:
    return [value * factor for value in x]


def adam_update(
    w: list[float], grad: list[float], m: list[float], v: list[float],
    grad_scale: float, beta1: float, beta2: float, decay: float,
    step_size: float, sqrt_correction: float, eps: float,
) -> tuple[list[float], list[float], list[float]]:
    g = [value * grad_scale for value in grad]
    m = [beta1 * m_i + (1.0 - beta1) * g_i for m_i, g_i in zip(m, g)]
    v = [beta2 * v_i + (1.0 - beta2) * g_i * g_i for v_i, g_i in zip(v, g)]
    new_w = [
        w_i * decay - step_size * m_i / (math.sqrt(v_i) / sqrt_correction + eps)
        for w_i, m_i, v_i in zip(w, m, v)
    ]
    return new_w, m, v


def sum_repeated(x: list[float], outer: int, size: int, inner: int) -> list[float]:
    # Source value s appears at positions (o * size + s) * inner + i.
    if inner == 1:
        return [sum(x[s::size]) for s in range(size)]
    return [
        sum(sum(x[(o * size + s) * inner : (o * size + s + 1) * inner]) for o in range(outer))
        for s in range(size)
    ]


def relu(x: list[float]) -> list[float]:
    return [value if value > 0.0 else 0.0 for value in x]


def relu_backward(x: list[float], grad: list[float]) -> list[float]:
    return [g if value > 0.0 else 0.0 for g, value in zip(grad, x)]


def tanh(x: list[float]) -> list[float]:
    return [math.tanh(value) for value in x]


def tanh_backward(y: list[float], grad: list[float]) -> list[float]:
    return [g * (1.0 - value * value) for g, value in zip(grad, y)]


def window_mean(
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


def window_mean_backward(
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


def window_max(
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

