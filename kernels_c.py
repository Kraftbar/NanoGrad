"""The C kernels: build ``kernels.c`` and call it through ``ctypes``.

``load()`` compiles ``kernels.c`` with the system C compiler (``$CC`` or
``cc``) the first time it is needed and caches the library under
``~/.cache/nanograd/``. The functions below have the same names and
arguments as their Python twins in ``kernels_reference.py``. They take any
flat sequence of floats (a C array of the current precision passes through
without a copy) and return C arrays: ``array("d")`` (float64), or
``array("f")`` after ``set_precision("f")``, which loads a second copy of
the kernels built with ``-DREAL=float``.

This file is plumbing: the math is in ``kernels.c`` and, readably, in
``kernels_reference.py``.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import tempfile
from array import array
from collections.abc import Sequence
from pathlib import Path


SOURCE = Path(__file__).resolve().parent / "kernels.c"

library: ctypes.CDLL | None = None     # the library for the current precision
typecode = "d"                          # "d": float64, "f": float32
_libraries: dict[str, ctypes.CDLL] = {}
_C_TYPES = {"d": ctypes.c_double, "f": ctypes.c_float}
_ZEROS = {"d": array("d", [0.0]), "f": array("f", [0.0])}


def load() -> bool:
    """Build (if needed) and load the C library; False if that fails."""

    return set_precision(typecode)


def set_precision(code: str) -> bool:
    """Use the float64 ("d") or float32 ("f") kernels; False if they fail to build."""

    global library, typecode
    if code not in _C_TYPES:
        raise ValueError("precision must be 'd' (float64) or 'f' (float32)")
    if code not in _libraries:
        built = _build_library(code)
        if built is None:
            return False
        _libraries[code] = built
        _keep_freed_memory()
    library, typecode = _libraries[code], code
    return True


def _keep_freed_memory() -> None:
    """Let freed tensor buffers be reused instead of handed back to the OS.

    Tensors are often a few megabytes. By default glibc returns each such
    block to the operating system when it is freed, so the next allocation
    has to fault its memory back in page by page: about a fifth of a small
    ResNet's training step. Raising glibc's thresholds keeps those blocks
    in the heap for reuse (PyTorch's caching allocator exists for the same
    reason). The cost is that the process keeps its peak memory. Does
    nothing on systems without glibc's ``mallopt``.
    """

    try:
        mallopt = ctypes.CDLL(None).mallopt
    except (OSError, AttributeError):
        return
    m_trim_threshold, m_mmap_threshold = -1, -3   # constants from <malloc.h>
    mallopt(m_mmap_threshold, 1 << 30)
    mallopt(m_trim_threshold, 1 << 30)




def _compile(compiler: str, flags: list[str]) -> Path | None:
    """Build kernels.c with ``flags`` into the cache (once), or None if that fails."""

    source = SOURCE.read_bytes()
    key = source + " ".join(flags).encode() + _cpu_signature().encode()
    digest = hashlib.sha256(key).hexdigest()[:16]
    cache = Path(os.environ.get("NANOGRAD_CACHE", Path.home() / ".cache" / "nanograd"))
    library_path = cache / f"kernels-{digest}.so"
    if library_path.exists():
        return library_path
    partial = None
    try:
        cache.mkdir(parents=True, exist_ok=True)
        # Build into a temporary name, then rename: two processes building
        # at once never see a half-written library.
        with tempfile.NamedTemporaryFile(dir=cache, suffix=".so", delete=False) as handle:
            partial = Path(handle.name)
        subprocess.run(
            [compiler, *flags, "-shared", "-fPIC", str(SOURCE), "-o", str(partial), "-lm"],
            check=True,
            capture_output=True,
        )
        partial.replace(library_path)
        return library_path
    except (OSError, subprocess.CalledProcessError):
        if partial is not None:
            partial.unlink(missing_ok=True)
        return None


def _cpu_signature() -> str:
    """The CPU model and instruction-set flags (Linux), else the platform name."""

    try:
        lines = Path("/proc/cpuinfo").read_text().splitlines()
    except OSError:
        return platform.machine() + platform.processor()
    wanted = [line for line in lines if line.startswith(("model name", "flags", "Features"))]
    return "\n".join(dict.fromkeys(wanted))


def _build_library(code: str) -> ctypes.CDLL | None:
    """Compile kernels.c for one precision (once per source version) and load it."""

    compiler = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if compiler is None or not SOURCE.exists():
        return None

    # By default the library is built for this exact CPU (-march=native), so
    # the compiler may use its widest vector instructions; kernels.c has a
    # fast matrix-product path for those. It is compiled on the machine that
    # runs it, and the CPU's identity is part of the cache key, so a shared
    # home directory never loads a library built for another CPU. If the
    # compiler rejects the flag, a portable build is used instead.
    # NANOGRAD_CFLAGS replaces "-march=native" (set it empty for a portable build).
    extra = os.environ.get("NANOGRAD_CFLAGS")
    attempts = [extra.split()] if extra is not None else [["-march=native"], []]
    real = "-DREAL=double" if code == "d" else "-DREAL=float"
    for flags in attempts:
        library_path = _compile(compiler, ["-O3", real, *flags])
        if library_path is not None:
            break
    else:
        return None

    try:
        library = ctypes.CDLL(str(library_path))
    except OSError:
        return None
    pointer = ctypes.POINTER(_C_TYPES[code])
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
    library.conv_forward.argtypes = [pointer, index_pointer, pointer, pointer] + [count, count] + [ctypes.c_int] * 3
    library.conv_kernel_grad.argtypes = [pointer, index_pointer, pointer, pointer] + [count, count] + [ctypes.c_int] * 3
    library.conv_input_grad.argtypes = [pointer, pointer, index_pointer, pointer] + [count, count] + [ctypes.c_int] * 3
    for name in ("gather_indexed", "scatter_add_indexed"):
        getattr(library, name).argtypes = [pointer, index_pointer, pointer, count, count, count]
    for name in ("gather_rows", "scatter_add_rows"):
        getattr(library, name).argtypes = [pointer, index_pointer, pointer, count, count]
    library.add_vectors.argtypes = [pointer, pointer, pointer, count]
    library.multiply_vectors.argtypes = [pointer, pointer, pointer, count]
    library.scale_vector.argtypes = [pointer, ctypes.c_double, pointer, count]
    library.add_scaled.argtypes = [pointer, pointer, ctypes.c_double, pointer, count]
    library.sum_of_squares.argtypes = [pointer, count]
    library.sum_of_squares.restype = ctypes.c_double
    library.adam_update.argtypes = [pointer] * 5 + [count] + [ctypes.c_double] * 7
    library.tanh_forward.argtypes = [pointer, pointer, count]
    library.relu_forward.argtypes = [pointer, pointer, count]
    library.bce_with_logits_sum.argtypes = [pointer, pointer, count]
    library.bce_with_logits_sum.restype = ctypes.c_double
    library.sigmoid_forward.argtypes = [pointer, pointer, count]
    library.sigmoid_backward.argtypes = [pointer, pointer, pointer, count]
    library.batch_norm_channels.argtypes = [pointer] * 8 + [count] * 3 + [ctypes.c_double]
    library.batch_norm_channels_backward.argtypes = [pointer] * 7 + [count] * 3
    library.rms_norm_rows.argtypes = [pointer] * 4 + [count, count, ctypes.c_double]
    library.rms_norm_rows_backward.argtypes = [pointer] * 6 + [count, count]
    library.sum_repeated.argtypes = [pointer, pointer, count, count, count]
    library.relu_backward.argtypes = [pointer, pointer, pointer, count]
    library.tanh_backward.argtypes = [pointer, pointer, pointer, count]
    for name in ("window_mean", "window_mean_backward"):
        getattr(library, name).argtypes = [pointer, index_pointer, pointer] + [count] * 4
    library.window_max.argtypes = [pointer, index_pointer, pointer, index_pointer] + [count] * 4
    return library


def _buffer(values: array):
    return (_C_TYPES[values.typecode] * len(values)).from_buffer(values)


def _reals(values: Sequence[float]) -> array:
    """The values as a C array of the current precision (passed through if it is one)."""

    return values if type(values) is array and values.typecode == typecode else array(typecode, values)


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
) -> array:
    size = c_step * (batches - 1) + m * n if c_step else m * n
    a_buffer = _reals(a)
    b_buffer = _reals(b)
    c_buffer = zeros(size)
    library.gemm(
        _buffer(a_buffer), _buffer(b_buffer), _buffer(c_buffer),
        batches, m, k, n,
        a_step, b_step, c_step,
        int(trans_a), int(trans_b),
    )
    return c_buffer


def zeros(count: int) -> array:
    return _ZEROS[typecode] * count   # repeating a one-element array is a fast fill


def gelu(x: list[float]) -> tuple[array, array]:
    x_buffer, y, cdf = _reals(x), zeros(len(x)), zeros(len(x))
    library.gelu_forward(_buffer(x_buffer), _buffer(y), _buffer(cdf), len(x))
    return y, cdf


def gelu_backward(x: list[float], cdf: list[float], grad: list[float]) -> array:
    dx = zeros(len(x))
    buffers = [_reals(values) for values in (x, cdf, grad)]
    library.gelu_backward(*map(_buffer, buffers), _buffer(dx), len(x))
    return dx


def softmax_rows(x: list[float], width: int) -> array:
    x_buffer, y = _reals(x), zeros(len(x))
    library.softmax_rows(_buffer(x_buffer), _buffer(y), len(x) // width, width)
    return y


def softmax_rows_backward(y: list[float], grad: list[float], width: int) -> array:
    y_buffer, grad_buffer, dx = _reals(y), _reals(grad), zeros(len(y))
    library.softmax_rows_backward(
        _buffer(y_buffer), _buffer(grad_buffer), _buffer(dx), len(y) // width, width,
    )
    return dx


def layer_norm_rows(
    x: list[float], weight: list[float], bias: list[float], width: int, eps: float,
) -> tuple[array, array, array]:
    rows = len(x) // width
    inputs = [_reals(values) for values in (x, weight, bias)]
    y, x_hat, inv_std = zeros(len(x)), zeros(len(x)), zeros(rows)
    library.layer_norm_rows(
        *map(_buffer, inputs), _buffer(y), _buffer(x_hat), _buffer(inv_std), rows, width, eps,
    )
    return y, x_hat, inv_std


def layer_norm_rows_backward(
    grad: list[float], x_hat: list[float], inv_std: list[float], weight: list[float],
    width: int, need_dx: bool,
) -> tuple[array | None, array, array]:
    inputs = [_reals(values) for values in (grad, x_hat, inv_std, weight)]
    dx = zeros(len(grad) if need_dx else 1)
    dweight, dbias = zeros(width), zeros(width)
    library.layer_norm_rows_backward(
        *map(_buffer, inputs), _buffer(dx), _buffer(dweight), _buffer(dbias),
        len(inv_std), width, int(need_dx),
    )
    return (dx if need_dx else None), dweight, dbias


def rms_norm_rows(x, weight, width: int, eps: float) -> tuple[array, array]:
    rows = len(x) // width
    x_buffer, weight_buffer = _reals(x), _reals(weight)
    y, inv_rms = zeros(len(x)), zeros(rows)
    library.rms_norm_rows(_buffer(x_buffer), _buffer(weight_buffer), _buffer(y), _buffer(inv_rms), rows, width, eps)
    return y, inv_rms


def rms_norm_rows_backward(grad, x, inv_rms, weight, width: int) -> tuple[array, array]:
    buffers = [_reals(values) for values in (grad, x, inv_rms, weight)]
    dx, dweight = zeros(len(x)), zeros(width)
    library.rms_norm_rows_backward(*map(_buffer, buffers), _buffer(dx), _buffer(dweight), len(inv_rms), width)
    return dx, dweight


def batch_norm_channels(x, weight, bias, batch: int, channels: int, plane: int, eps: float):
    inputs = [_reals(values) for values in (x, weight, bias)]
    y, x_hat = zeros(len(x)), zeros(len(x))
    inv_std, mean, variance = zeros(channels), zeros(channels), zeros(channels)
    library.batch_norm_channels(
        *map(_buffer, inputs), _buffer(y), _buffer(x_hat), _buffer(inv_std), _buffer(mean), _buffer(variance),
        batch, channels, plane, eps,
    )
    return y, x_hat, inv_std, mean, variance


def batch_norm_channels_backward(grad, x_hat, inv_std, weight, batch: int, channels: int, plane: int):
    inputs = [_reals(values) for values in (grad, x_hat, inv_std, weight)]
    dx, dweight, dbias = zeros(len(grad)), zeros(channels), zeros(channels)
    library.batch_norm_channels_backward(
        *map(_buffer, inputs), _buffer(dx), _buffer(dweight), _buffer(dbias), batch, channels, plane,
    )
    return dx, dweight, dbias


def _index_buffer(index: Sequence[int]):
    is_table = type(index) is array and index.typecode == "l"   # see index_table
    values = index if is_table else array("l", index)
    return values, (ctypes.c_long * len(values)).from_buffer(values)


def conv_forward(image, index, kernel, batches: int, image_size: int,
                 filters: int, patch_size: int, positions: int) -> array:
    image_buffer, kernel_buffer = _reals(image), _reals(kernel)
    keep, index_buffer = _index_buffer(index)
    out = zeros(batches * filters * positions)
    library.conv_forward(_buffer(image_buffer), index_buffer, _buffer(kernel_buffer), _buffer(out),
                         batches, image_size, filters, patch_size, positions)
    return out


def conv_kernel_grad(image, index, grad, batches: int, image_size: int,
                     filters: int, patch_size: int, positions: int) -> array:
    image_buffer, grad_buffer = _reals(image), _reals(grad)
    keep, index_buffer = _index_buffer(index)
    dkernel = zeros(filters * patch_size)
    library.conv_kernel_grad(_buffer(image_buffer), index_buffer, _buffer(grad_buffer), _buffer(dkernel),
                             batches, image_size, filters, patch_size, positions)
    return dkernel


def conv_input_grad(grad, kernel, index, batches: int, image_size: int,
                    filters: int, patch_size: int, positions: int) -> array:
    grad_buffer, kernel_buffer = _reals(grad), _reals(kernel)
    keep, index_buffer = _index_buffer(index)
    image_grad = zeros(batches * image_size)
    library.conv_input_grad(_buffer(grad_buffer), _buffer(kernel_buffer), index_buffer, _buffer(image_grad),
                            batches, image_size, filters, patch_size, positions)
    return image_grad


def gather_indexed(
    image: list[float], index: Sequence[int], batches: int, image_size: int,
) -> array:
    image_buffer = _reals(image)
    keep, index_buffer = _index_buffer(index)
    out = zeros(batches * len(index))
    library.gather_indexed(
        _buffer(image_buffer), index_buffer, _buffer(out), batches, image_size, len(index),
    )
    return out


def scatter_add_indexed(
    grad: list[float], index: Sequence[int], batches: int, image_size: int, into: array | None = None,
) -> array:
    grad_buffer = _reals(grad)
    keep, index_buffer = _index_buffer(index)
    if into is not None and (type(into) is not array or into.typecode != typecode):
        raise TypeError("scatter_add_indexed(into=...) needs a buffer from zeros() in the current precision")
    image_grad = zeros(batches * image_size) if into is None else into
    library.scatter_add_indexed(
        _buffer(grad_buffer), index_buffer, _buffer(image_grad), batches, image_size, len(index),
    )
    return image_grad


def gather_rows(table: list[float], ids: Sequence[int], width: int) -> array:
    table_buffer = _reals(table)
    keep, id_buffer = _index_buffer(ids)
    out = zeros(len(ids) * width)
    library.gather_rows(_buffer(table_buffer), id_buffer, _buffer(out), len(ids), width)
    return out


def scatter_add_rows(grad: list[float], ids: Sequence[int], width: int, rows: int) -> array:
    grad_buffer = _reals(grad)
    keep, id_buffer = _index_buffer(ids)
    table_grad = zeros(rows * width)
    library.scatter_add_rows(_buffer(grad_buffer), id_buffer, _buffer(table_grad), len(ids), width)
    return table_grad


def _elementwise(function, x: list[float], y: list[float]) -> array:
    if len(x) != len(y):
        raise ValueError("elementwise kernels need equal lengths")
    x_buffer, y_buffer, out = _reals(x), _reals(y), zeros(len(x))
    function(_buffer(x_buffer), _buffer(y_buffer), _buffer(out), len(x))
    return out


def add(x: list[float], y: list[float]) -> array:
    return _elementwise(library.add_vectors, x, y)


def multiply(x: list[float], y: list[float]) -> array:
    return _elementwise(library.multiply_vectors, x, y)


def scale(x: list[float], factor: float) -> array:
    x_buffer, out = _reals(x), zeros(len(x))
    library.scale_vector(_buffer(x_buffer), factor, _buffer(out), len(x))
    return out


def add_scaled(x: list[float], y: list[float], factor: float) -> array:
    if len(x) != len(y):
        raise ValueError("add_scaled needs equal lengths")
    x_buffer, y_buffer, out = _reals(x), _reals(y), zeros(len(x))
    library.add_scaled(_buffer(x_buffer), _buffer(y_buffer), factor, _buffer(out), len(x))
    return out


def sum_of_squares(x: list[float]) -> float:
    return library.sum_of_squares(_buffer(_reals(x)), len(x))


def adam_update(w, grad, m, v, *hyperparameters) -> tuple[array, array, array]:
    # m and v are the optimizer's own state: updated in place once they are
    # C arrays (the first call converts them). The weights get a new array.
    if not len(w) == len(grad) == len(m) == len(v):
        raise ValueError("adam_update needs weights, gradient and moments of equal length")
    buffers = [_reals(values) for values in (w, grad, m, v)]
    new_w = zeros(len(w))
    library.adam_update(*map(_buffer, buffers), _buffer(new_w), len(w), *hyperparameters)
    return new_w, buffers[2], buffers[3]


def sum_repeated(x: list[float], outer: int, size: int, inner: int) -> array:
    if len(x) != outer * size * inner:
        raise ValueError("sum_repeated needs outer * size * inner values")
    x_buffer, out = _reals(x), zeros(size)
    library.sum_repeated(_buffer(x_buffer), _buffer(out), outer, size, inner)
    return out


def sigmoid(x: list[float]) -> array:
    x_buffer, y = _reals(x), zeros(len(x))
    library.sigmoid_forward(_buffer(x_buffer), _buffer(y), len(x))
    return y


def sigmoid_backward(y: list[float], grad: list[float]) -> array:
    y_buffer, grad_buffer, dx = _reals(y), _reals(grad), zeros(len(y))
    library.sigmoid_backward(_buffer(y_buffer), _buffer(grad_buffer), _buffer(dx), len(y))
    return dx


def bce_with_logits_sum(x: list[float], t: list[float]) -> float:
    if len(x) != len(t):
        raise ValueError("bce_with_logits_sum needs equal lengths")
    x_buffer, t_buffer = _reals(x), _reals(t)
    return library.bce_with_logits_sum(_buffer(x_buffer), _buffer(t_buffer), len(x))


def relu(x: list[float]) -> array:
    x_buffer, y = _reals(x), zeros(len(x))
    library.relu_forward(_buffer(x_buffer), _buffer(y), len(x))
    return y


def relu_backward(x: list[float], grad: list[float]) -> array:
    x_buffer, grad_buffer, dx = _reals(x), _reals(grad), zeros(len(x))
    library.relu_backward(_buffer(x_buffer), _buffer(grad_buffer), _buffer(dx), len(x))
    return dx


def tanh(x: list[float]) -> array:
    x_buffer, y = _reals(x), zeros(len(x))
    library.tanh_forward(_buffer(x_buffer), _buffer(y), len(x))
    return y


def tanh_backward(y: list[float], grad: list[float]) -> array:
    y_buffer, grad_buffer, dx = _reals(y), _reals(grad), zeros(len(y))
    library.tanh_backward(_buffer(y_buffer), _buffer(grad_buffer), _buffer(dx), len(y))
    return dx


def window_mean(
    image: list[float], index: Sequence[int], planes: int, plane_size: int, window_size: int,
) -> array:
    image_buffer = _reals(image)
    keep, index_buffer = _index_buffer(index)
    windows = len(index) // window_size
    out = zeros(planes * windows)
    library.window_mean(
        _buffer(image_buffer), index_buffer, _buffer(out), planes, plane_size, windows, window_size,
    )
    return out


def window_mean_backward(
    grad: list[float], index: Sequence[int], planes: int, plane_size: int, window_size: int,
) -> array:
    grad_buffer = _reals(grad)
    keep, index_buffer = _index_buffer(index)
    image_grad = zeros(planes * plane_size)
    library.window_mean_backward(
        _buffer(grad_buffer), index_buffer, _buffer(image_grad),
        planes, plane_size, len(index) // window_size, window_size,
    )
    return image_grad


def window_max(
    image: list[float], index: Sequence[int], planes: int, plane_size: int, window_size: int,
) -> tuple[list[float], list[int]]:
    image_buffer = _reals(image)
    keep, index_buffer = _index_buffer(index)
    windows = len(index) // window_size
    out = zeros(planes * windows)
    argmax, argmax_buffer = _index_buffer([0] * (planes * windows))
    library.window_max(
        _buffer(image_buffer), index_buffer, _buffer(out), argmax_buffer,
        planes, plane_size, windows, window_size,
    )
    return out, argmax
