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


def rms_norm_rows(
    x: list[float], weight: list[float], width: int, eps: float,
) -> tuple[list[float], list[float]]:
    y: list[float] = []
    inv_rms: list[float] = []
    for start in range(0, len(x), width):
        row = x[start : start + width]
        scale = 1.0 / math.sqrt(sum(value * value for value in row) / width + eps)
        inv_rms.append(scale)
        y.extend([value * scale * w for value, w in zip(row, weight)])
    return y, inv_rms


def rms_norm_rows_backward(
    grad: list[float], x: list[float], inv_rms: list[float], weight: list[float], width: int,
) -> tuple[list[float], list[float]]:
    dx: list[float] = []
    dweight = [0.0] * width
    for row, scale in enumerate(inv_rms):
        dy = grad[row * width : (row + 1) * width]
        values = x[row * width : (row + 1) * width]
        dweight = [total + g * v * scale for total, g, v in zip(dweight, dy, values)]
        mean_g_x = sum(g * w * v for g, w, v in zip(dy, weight, values)) / width
        dx.extend([scale * (g * w - v * scale * scale * mean_g_x) for g, w, v in zip(dy, weight, values)])
    return dx, dweight


def batch_norm_channels(
    x: list[float], weight: list[float], bias: list[float],
    batch: int, channels: int, plane: int, eps: float,
):
    """Returns (y, x_hat, inv_std, mean, variance), the last three per channel."""

    count = batch * plane
    y, x_hat = [0.0] * len(x), [0.0] * len(x)
    inv_std, mean, variance = [], [], []
    for c in range(channels):
        blocks = [range((b * channels + c) * plane, (b * channels + c + 1) * plane) for b in range(batch)]
        values = [x[i] for block in blocks for i in block]
        m = sum(values) / count
        v = sum((value - m) ** 2 for value in values) / count
        scale = 1.0 / math.sqrt(v + eps)
        mean.append(m)
        variance.append(v)
        inv_std.append(scale)
        for block in blocks:
            for i in block:
                x_hat[i] = (x[i] - m) * scale
                y[i] = x_hat[i] * weight[c] + bias[c]
    return y, x_hat, inv_std, mean, variance


def batch_norm_channels_backward(
    grad: list[float], x_hat: list[float], inv_std: list[float], weight: list[float],
    batch: int, channels: int, plane: int,
):
    """Returns (dx, dweight, dbias)."""

    count = batch * plane
    dx = [0.0] * len(grad)
    dweight, dbias = [], []
    for c in range(channels):
        indices = [i for b in range(batch) for i in range((b * channels + c) * plane, (b * channels + c + 1) * plane)]
        sum_dy = sum(grad[i] for i in indices)
        sum_dy_h = sum(grad[i] * x_hat[i] for i in indices)
        dweight.append(sum_dy_h)
        dbias.append(sum_dy)
        mean_g, mean_g_h = weight[c] * sum_dy / count, weight[c] * sum_dy_h / count
        for i in indices:
            dx[i] = inv_std[c] * (grad[i] * weight[c] - mean_g - x_hat[i] * mean_g_h)
    return dx, dweight, dbias


def conv_forward(
    image: list[float], index: Sequence[int], kernel: list[float],
    batches: int, image_size: int, filters: int, patch_size: int, positions: int,
) -> list[float]:
    """out_b (filters x positions) = kernel @ patches_b^T, reading each patch
    straight from the image: patch j is image_b[index[j * K : (j + 1) * K]]."""

    out: list[float] = []
    rows = [kernel[f * patch_size : (f + 1) * patch_size] for f in range(filters)]
    for batch in range(batches):
        image_b = image[batch * image_size : (batch + 1) * image_size]
        patches = [
            [image_b[i] for i in index[j * patch_size : (j + 1) * patch_size]]
            for j in range(positions)
        ]
        for row in rows:
            out.extend([dot(row, patch) for patch in patches])
    return out


def conv_kernel_grad(
    image: list[float], index: Sequence[int], grad: list[float],
    batches: int, image_size: int, filters: int, patch_size: int, positions: int,
) -> list[float]:
    """dkernel = sum over images of grad_b (filters x positions) @ patches_b."""

    dkernel = [0.0] * (filters * patch_size)
    for batch in range(batches):
        image_b = image[batch * image_size : (batch + 1) * image_size]
        grad_b = grad[batch * filters * positions : (batch + 1) * filters * positions]
        for f in range(filters):
            for q in range(patch_size):
                column = [image_b[index[p * patch_size + q]] for p in range(positions)]
                dkernel[f * patch_size + q] += dot(grad_b[f * positions : (f + 1) * positions], column)
    return dkernel


def conv_input_grad(
    grad: list[float], kernel: list[float], index: Sequence[int],
    batches: int, image_size: int, filters: int, patch_size: int, positions: int,
) -> list[float]:
    """image_grad: each patch's gradient (grad_b^T @ kernel, one row per
    position) added back onto the pixels the patch was read from."""

    image_grad = [0.0] * (batches * image_size)
    columns = [kernel[q::patch_size] for q in range(patch_size)]          # kernel column q
    for batch in range(batches):
        grad_b = grad[batch * filters * positions : (batch + 1) * filters * positions]
        start = batch * image_size
        for j in range(positions):
            grad_column = grad_b[j::positions]                            # dOut[:, j]
            for q in range(patch_size):
                image_grad[start + index[j * patch_size + q]] += dot(grad_column, columns[q])
    return image_grad


def gather_indexed(
    image: list[float], index: Sequence[int], batches: int, image_size: int,
) -> list[float]:
    out: list[float] = []
    for batch in range(batches):
        start = batch * image_size
        single = image[start : start + image_size]
        out.extend([single[i] for i in index])
    return out


def gather_rows(table: list[float], ids: Sequence[int], width: int) -> list[float]:
    out: list[float] = []
    for row in ids:
        out.extend(table[row * width : (row + 1) * width])
    return out


def scatter_add_rows(grad: list[float], ids: Sequence[int], width: int, rows: int) -> list[float]:
    table_grad = [0.0] * (rows * width)
    for position, row in enumerate(ids):
        start = row * width
        table_grad[start : start + width] = [
            total + g for total, g in zip(table_grad[start : start + width], grad[position * width : (position + 1) * width])
        ]
    return table_grad


def scatter_add_indexed(
    grad: list[float], index: Sequence[int], batches: int, image_size: int, into: list[float] | None = None,
) -> list[float]:
    image_grad = [0.0] * (batches * image_size) if into is None else into
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


def dropout_mask(count: int, p: float, seed: int) -> list[float]:
    keep, mask, bits = 1.0 / (1.0 - p), [], (1 << 64) - 1
    for i in range(count):                                    # splitmix64, as in kernels.c
        z = (seed + (i + 1) * 0x9E3779B97F4A7C15) & bits
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & bits
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & bits
        z ^= z >> 31
        mask.append(0.0 if (z >> 11) * 2.0 ** -53 < p else keep)
    return mask


def add_scaled(x: list[float], y: list[float], factor: float) -> list[float]:
    return [a + factor * b for a, b in zip(x, y)]


def sum_of_squares(x: list[float]) -> float:
    return math.fsum(value * value for value in x)


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


def _sigmoid1(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def lstm_memory(gates, c, batch: int, hidden: int) -> list[float]:
    out = []
    for b in range(batch):
        row = gates[b * 4 * hidden : (b + 1) * 4 * hidden]
        for j in range(hidden):
            i, f, g = _sigmoid1(row[j]), _sigmoid1(row[hidden + j]), math.tanh(row[2 * hidden + j])
            out.append(f * c[b * hidden + j] + i * g)
    return out


def lstm_memory_backward(gates, c, grad, batch: int, hidden: int) -> tuple[list[float], list[float]]:
    dgates, dc = [], []
    for b in range(batch):
        row = gates[b * 4 * hidden : (b + 1) * 4 * hidden]
        di, df, dg = [], [], []
        for j in range(hidden):
            i, f, g = _sigmoid1(row[j]), _sigmoid1(row[hidden + j]), math.tanh(row[2 * hidden + j])
            d = grad[b * hidden + j]
            di.append(d * g * i * (1.0 - i))
            df.append(d * c[b * hidden + j] * f * (1.0 - f))
            dg.append(d * i * (1.0 - g * g))
            dc.append(d * f)
        dgates += di + df + dg + [0.0] * hidden
    return dgates, dc


def lstm_output(gates, c_new, batch: int, hidden: int) -> list[float]:
    return [
        _sigmoid1(gates[b * 4 * hidden + 3 * hidden + j]) * math.tanh(c_new[b * hidden + j])
        for b in range(batch) for j in range(hidden)
    ]


def lstm_output_backward(gates, c_new, grad, batch: int, hidden: int) -> tuple[list[float], list[float]]:
    dgates, dc_new = [], []
    for b in range(batch):
        do = []
        for j in range(hidden):
            o, t = _sigmoid1(gates[b * 4 * hidden + 3 * hidden + j]), math.tanh(c_new[b * hidden + j])
            d = grad[b * hidden + j]
            do.append(d * t * o * (1.0 - o))
            dc_new.append(d * o * (1.0 - t * t))
        dgates += [0.0] * (3 * hidden) + do
    return dgates, dc_new


def bce_with_logits_sum(x: list[float], t: list[float]) -> float:
    return math.fsum(max(v, 0.0) - v * target + math.log1p(math.exp(-abs(v))) for v, target in zip(x, t))


def relu(x: list[float]) -> list[float]:
    return [value if value > 0.0 else 0.0 for value in x]


def relu_backward(x: list[float], grad: list[float]) -> list[float]:
    return [g if value > 0.0 else 0.0 for g, value in zip(grad, x)]


def sigmoid(x: list[float]) -> list[float]:
    # Two algebraically equal forms, each safe from overflow on its own side.
    return [
        1.0 / (1.0 + math.exp(-value)) if value >= 0.0 else math.exp(value) / (1.0 + math.exp(value))
        for value in x
    ]


def sigmoid_backward(y: list[float], grad: list[float]) -> list[float]:
    return [g * value * (1.0 - value) for g, value in zip(grad, y)]


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

