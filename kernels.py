"""Compute kernels with two interchangeable backends: C and pure Python.

The heavy inner loops (for now, batched matrix multiplication) live here so
``tensor.py`` can stay about the math. Two backends compute the same thing:

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
                [compiler, "-O3", "-shared", "-fPIC", str(SOURCE), "-o", str(partial)],
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
    return library


def _c_buffer(values: array):
    return (ctypes.c_double * len(values)).from_buffer(values)


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
) -> list[float]:
    size = c_step * (batches - 1) + m * n if c_step else m * n
    a_buffer = array("d", a)
    b_buffer = array("d", b)
    c_buffer = array("d", bytes(8 * size))
    _library.gemm(
        _c_buffer(a_buffer), _c_buffer(b_buffer), _c_buffer(c_buffer),
        batches, m, k, n,
        a_step, b_step, c_step,
        int(trans_a), int(trans_b),
    )
    return c_buffer.tolist()


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
) -> list[float]:
    """Batched ``C = op(A) @ op(B)``; see ``kernels.c`` for the conventions.

    A ``c_step`` of 0 sums all batch products into one (m x n) result.
    """

    kernel = _c_gemm if BACKEND == "c" else _python_gemm
    return kernel(a, b, batches, m, k, n, a_step, b_step, c_step, trans_a, trans_b)
