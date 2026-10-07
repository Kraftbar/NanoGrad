"""The C kernels: build ``kernels.c`` and call it through ``ctypes``.

``load()`` compiles ``kernels.c`` with the system C compiler (``$CC`` or
``cc``) the first time it is needed and caches the library under
``~/.cache/nanograd/``. The functions below have the same names and
arguments as their Python twins in ``kernels_reference.py``. They take any
flat sequence of floats (a C array passes through without a copy) and
return C arrays of doubles (``array("d")``).

This file is plumbing: the math is in ``kernels.c`` and, readably, in
``kernels_reference.py``.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from array import array
from collections.abc import Sequence
from pathlib import Path


SOURCE = Path(__file__).resolve().parent / "kernels.c"

library: ctypes.CDLL | None = None


def load() -> bool:
    """Build (if needed) and load the C library; False if that fails."""

    global library
    if library is None:
        library = _build_library()
    return library is not None




def _build_library() -> ctypes.CDLL | None:
    """Compile kernels.c (once per source version) and load it, or None."""

    compiler = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if compiler is None or not SOURCE.exists():
        return None

    # NANOGRAD_CFLAGS adds compiler flags, e.g. "-march=native" to use every
    # vector instruction this CPU has (faster, but the library then only
    # runs on CPUs like this one). The flags are part of the cache key.
    flags = ["-O3", *os.environ.get("NANOGRAD_CFLAGS", "").split()]
    source = SOURCE.read_bytes()
    digest = hashlib.sha256(source + " ".join(flags).encode()).hexdigest()[:16]
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
                [compiler, *flags, "-shared", "-fPIC", str(SOURCE), "-o", str(partial), "-lm"],
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
    library.add_vectors.argtypes = [pointer, pointer, pointer, count]
    library.multiply_vectors.argtypes = [pointer, pointer, pointer, count]
    library.scale_vector.argtypes = [pointer, ctypes.c_double, pointer, count]
    library.adam_update.argtypes = [pointer] * 5 + [count] + [ctypes.c_double] * 7
    library.tanh_forward.argtypes = [pointer, pointer, count]
    library.tanh_backward.argtypes = [pointer, pointer, pointer, count]
    for name in ("window_mean", "window_mean_backward"):
        getattr(library, name).argtypes = [pointer, index_pointer, pointer] + [count] * 4
    library.window_max.argtypes = [pointer, index_pointer, pointer, index_pointer] + [count] * 4
    return library


def _buffer(values: array):
    return (ctypes.c_double * len(values)).from_buffer(values)


def _doubles(values: Sequence[float]) -> array:
    """The values as a C double array; an array passes through uncopied."""

    return values if isinstance(values, array) else array("d", values)


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
    a_buffer = _doubles(a)
    b_buffer = _doubles(b)
    c_buffer = zeros(size)
    library.gemm(
        _buffer(a_buffer), _buffer(b_buffer), _buffer(c_buffer),
        batches, m, k, n,
        a_step, b_step, c_step,
        int(trans_a), int(trans_b),
    )
    return c_buffer


_ZERO = array("d", [0.0])


def zeros(count: int) -> array:
    return _ZERO * count   # repeating a one-element array is a fast fill


def gelu(x: list[float]) -> tuple[array, array]:
    x_buffer, y, cdf = _doubles(x), zeros(len(x)), zeros(len(x))
    library.gelu_forward(_buffer(x_buffer), _buffer(y), _buffer(cdf), len(x))
    return y, cdf


def gelu_backward(x: list[float], cdf: list[float], grad: list[float]) -> array:
    dx = zeros(len(x))
    buffers = [_doubles(values) for values in (x, cdf, grad)]
    library.gelu_backward(*map(_buffer, buffers), _buffer(dx), len(x))
    return dx


def softmax_rows(x: list[float], width: int) -> array:
    x_buffer, y = _doubles(x), zeros(len(x))
    library.softmax_rows(_buffer(x_buffer), _buffer(y), len(x) // width, width)
    return y


def softmax_rows_backward(y: list[float], grad: list[float], width: int) -> array:
    y_buffer, grad_buffer, dx = _doubles(y), _doubles(grad), zeros(len(y))
    library.softmax_rows_backward(
        _buffer(y_buffer), _buffer(grad_buffer), _buffer(dx), len(y) // width, width,
    )
    return dx


def layer_norm_rows(
    x: list[float], weight: list[float], bias: list[float], width: int, eps: float,
) -> tuple[array, array, array]:
    rows = len(x) // width
    inputs = [_doubles(values) for values in (x, weight, bias)]
    y, x_hat, inv_std = zeros(len(x)), zeros(len(x)), zeros(rows)
    library.layer_norm_rows(
        *map(_buffer, inputs), _buffer(y), _buffer(x_hat), _buffer(inv_std), rows, width, eps,
    )
    return y, x_hat, inv_std


def layer_norm_rows_backward(
    grad: list[float], x_hat: list[float], inv_std: list[float], weight: list[float],
    width: int, need_dx: bool,
) -> tuple[array | None, array, array]:
    inputs = [_doubles(values) for values in (grad, x_hat, inv_std, weight)]
    dx = zeros(len(grad) if need_dx else 1)
    dweight, dbias = zeros(width), zeros(width)
    library.layer_norm_rows_backward(
        *map(_buffer, inputs), _buffer(dx), _buffer(dweight), _buffer(dbias),
        len(inv_std), width, int(need_dx),
    )
    return (dx if need_dx else None), dweight, dbias


def _index_buffer(index: Sequence[int]):
    is_table = type(index) is array and index.typecode == "l"   # see index_table
    values = index if is_table else array("l", index)
    return values, (ctypes.c_long * len(values)).from_buffer(values)


def gather_indexed(
    image: list[float], index: Sequence[int], batches: int, image_size: int,
) -> array:
    image_buffer = _doubles(image)
    keep, index_buffer = _index_buffer(index)
    out = zeros(batches * len(index))
    library.gather_indexed(
        _buffer(image_buffer), index_buffer, _buffer(out), batches, image_size, len(index),
    )
    return out


def scatter_add_indexed(
    grad: list[float], index: Sequence[int], batches: int, image_size: int,
) -> array:
    grad_buffer = _doubles(grad)
    keep, index_buffer = _index_buffer(index)
    image_grad = zeros(batches * image_size)
    library.scatter_add_indexed(
        _buffer(grad_buffer), index_buffer, _buffer(image_grad), batches, image_size, len(index),
    )
    return image_grad


def _elementwise(function, x: list[float], y: list[float]) -> array:
    if len(x) != len(y):
        raise ValueError("elementwise kernels need equal lengths")
    x_buffer, y_buffer, out = _doubles(x), _doubles(y), zeros(len(x))
    function(_buffer(x_buffer), _buffer(y_buffer), _buffer(out), len(x))
    return out


def add(x: list[float], y: list[float]) -> array:
    return _elementwise(library.add_vectors, x, y)


def multiply(x: list[float], y: list[float]) -> array:
    return _elementwise(library.multiply_vectors, x, y)


def scale(x: list[float], factor: float) -> array:
    x_buffer, out = _doubles(x), zeros(len(x))
    library.scale_vector(_buffer(x_buffer), factor, _buffer(out), len(x))
    return out


def adam_update(w, grad, m, v, *hyperparameters) -> tuple[array, array, array]:
    # m and v are the optimizer's own state: updated in place once they are
    # C arrays (the first call converts them). The weights get a new array.
    buffers = [_doubles(values) for values in (w, grad, m, v)]
    new_w = zeros(len(w))
    library.adam_update(*map(_buffer, buffers), _buffer(new_w), len(w), *hyperparameters)
    return new_w, buffers[2], buffers[3]


def tanh(x: list[float]) -> array:
    x_buffer, y = _doubles(x), zeros(len(x))
    library.tanh_forward(_buffer(x_buffer), _buffer(y), len(x))
    return y


def tanh_backward(y: list[float], grad: list[float]) -> array:
    y_buffer, grad_buffer, dx = _doubles(y), _doubles(grad), zeros(len(y))
    library.tanh_backward(_buffer(y_buffer), _buffer(grad_buffer), _buffer(dx), len(y))
    return dx


def window_mean(
    image: list[float], index: Sequence[int], planes: int, plane_size: int, window_size: int,
) -> array:
    image_buffer = _doubles(image)
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
    grad_buffer = _doubles(grad)
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
    image_buffer = _doubles(image)
    keep, index_buffer = _index_buffer(index)
    windows = len(index) // window_size
    out = zeros(planes * windows)
    argmax, argmax_buffer = _index_buffer([0] * (planes * windows))
    library.window_max(
        _buffer(image_buffer), index_buffer, _buffer(out), argmax_buffer,
        planes, plane_size, windows, window_size,
    )
    return out, argmax
