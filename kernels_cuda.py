"""The CUDA kernels: compile ``kernels.cu`` and call it through ``ctypes``.

The GPU twin of ``kernels_c.py``. ``load()`` compiles ``kernels.cu`` with
NVRTC, NVIDIA's runtime compiler (the ``nvidia-cuda-nvrtc`` pip package),
for this exact GPU the first time it is needed, caches the binary under
``~/.cache/nanograd/``, and loads it through the CUDA driver (``libcuda``,
part of the NVIDIA driver). Nothing else is needed: no CUDA toolkit, no
build step.

Tensors keep their values in GPU memory, in a ``DeviceArray``. Kernels read
and write DeviceArrays, so data stays on the GPU from op to op. A
DeviceArray is also a flat sequence of floats like ``array("d")``: indexing,
iterating or ``sum()`` copies it to the host once (a *sync*; ``syncs``
counts them) and keeps that copy until a kernel changes the values. So any
code written for the other backends runs unchanged, and the slow spots
show up as syncs.

The functions below have the same names and arguments as their twins in
``kernels_c.py``. A kernel that has no CUDA version yet runs the C one on
host copies (see ``missing()``): slow, but correct, so whole models run
while kernels are added one by one.

This file is plumbing: the math is in ``kernels.cu`` and, readably, in
``kernels_reference.py``.
"""

from __future__ import annotations

import ctypes
import glob
import hashlib
import os
import sys
from array import array
from collections.abc import Sequence
from pathlib import Path

import kernels_c

SOURCE = Path(__file__).resolve().parent / "kernels.cu"

typecode = "d"                    # "d": float64, "f": float32 (what new tensors are stored as)
load_error = ""                   # why load() failed, if it did
syncs = 0                         # copies from the GPU to the host so far
_driver = None                    # libcuda
_modules: dict[str, ctypes.c_void_p] = {}       # compiled kernels.cu, one per precision
_functions: dict[tuple[str, str], ctypes.c_void_p] = {}
_ITEM_SIZE = {"d": 8, "f": 4, "l": 8}
BLOCK = 256                       # threads per block for the one-thread-per-value kernels


# ---------------------------------------------------------------------------
# The driver and the compiler
# ---------------------------------------------------------------------------


def _check(result: int, call: str = "") -> None:
    if result != 0:
        name = ctypes.c_char_p()
        _driver.cuGetErrorName(result, ctypes.byref(name))
        raise RuntimeError(f"CUDA error {result} ({name.value.decode() if name.value else '?'}) in {call}")


def _find_nvrtc() -> str:
    """libnvrtc from the nvidia-cuda-nvrtc wheel (any CUDA version), else the system's."""

    for folder in sys.path:
        found = sorted(glob.glob(os.path.join(folder, "nvidia", "cu*", "lib", "libnvrtc.so.*"))
                       + glob.glob(os.path.join(folder, "nvidia", "cuda_nvrtc", "lib", "libnvrtc.so.*")))
        found = [path for path in found if "builtins" not in path]
        if found:
            return found[-1]
    return "libnvrtc.so"


def _compile(code: str, arch: str) -> bytes:
    """kernels.cu for one precision and GPU architecture, as a cubin (cached on disk)."""

    options = [f"--gpu-architecture={arch}", f"-DREAL={'double' if code == 'd' else 'float'}",
               "-default-device", "--std=c++17"]
    source = SOURCE.read_bytes()
    digest = hashlib.sha256(source + " ".join(options).encode()).hexdigest()[:16]
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "nanograd"
    cached = cache / f"kernels_cuda_{digest}.cubin"
    if cached.exists():
        return cached.read_bytes()

    nvrtc = ctypes.CDLL(_find_nvrtc())
    program = ctypes.c_void_p()
    if nvrtc.nvrtcCreateProgram(ctypes.byref(program), source, b"kernels.cu", 0, None, None):
        raise RuntimeError("nvrtcCreateProgram failed")
    flags = (ctypes.c_char_p * len(options))(*[o.encode() for o in options])
    status = nvrtc.nvrtcCompileProgram(program, len(options), flags)
    size = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
    log = ctypes.create_string_buffer(size.value)
    nvrtc.nvrtcGetProgramLog(program, log)
    if status != 0:
        raise RuntimeError("kernels.cu did not compile:\n" + log.value.decode())
    nvrtc.nvrtcGetCUBINSize(program, ctypes.byref(size))
    cubin = ctypes.create_string_buffer(size.value)
    nvrtc.nvrtcGetCUBIN(program, cubin)
    nvrtc.nvrtcDestroyProgram(ctypes.byref(program))
    cache.mkdir(parents=True, exist_ok=True)
    temporary = cached.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_bytes(cubin.raw)
    temporary.replace(cached)                   # atomic: parallel processes never read half a file
    return cubin.raw


def load() -> bool:
    """Start the driver and compile/load kernels.cu (once); False if that fails (see load_error)."""

    global _driver, load_error
    if typecode in _modules:
        return True
    try:
        if _driver is None:
            driver = ctypes.CDLL("libcuda.so.1")
            _driver = driver
            _check(driver.cuInit(0), "cuInit")
            device, context = ctypes.c_int(), ctypes.c_void_p()
            _check(driver.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
            _check(driver.cuDevicePrimaryCtxRetain(ctypes.byref(context), device), "cuDevicePrimaryCtxRetain")
            _check(driver.cuCtxSetCurrent(context), "cuCtxSetCurrent")
            driver.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
            driver.cuMemFree_v2.argtypes = [ctypes.c_uint64]
            driver.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
            driver.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
            driver.cuMemcpyDtoDAsync_v2.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p]
            driver.cuMemsetD8Async.argtypes = [ctypes.c_uint64, ctypes.c_ubyte, ctypes.c_size_t, ctypes.c_void_p]
            driver.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        major, minor = ctypes.c_int(), ctypes.c_int()
        _check(_driver.cuDeviceGetAttribute(ctypes.byref(major), 75, 0), "compute capability")   # 75, 76: major, minor
        _check(_driver.cuDeviceGetAttribute(ctypes.byref(minor), 76, 0), "compute capability")
        cubin = _compile(typecode, f"sm_{major.value}{minor.value}")
        module = ctypes.c_void_p()
        _check(_driver.cuModuleLoadData(ctypes.byref(module), cubin), "cuModuleLoadData")
        _modules[typecode] = module
        return True
    except (OSError, RuntimeError) as error:
        load_error = str(error)
        return False


def set_precision(code: str) -> None:
    """Store new tensors as float64 ("d") or float32 ("f")."""

    global typecode
    if code not in ("d", "f"):
        raise ValueError("precision must be 'd' (float64) or 'f' (float32)")
    typecode = code
    if not load():
        raise RuntimeError(f"the CUDA kernels could not be loaded: {load_error}")


def _function(name: str, code: str):
    key = (name, code)
    if key not in _functions:
        function = ctypes.c_void_p()
        _check(_driver.cuModuleGetFunction(ctypes.byref(function), _modules[code], name.encode()),
               f"cuModuleGetFunction({name})")
        _functions[key] = function
    return _functions[key]


def launch(name: str, code: str, grid, block, *args, shared: int = 0) -> None:
    """Run kernel ``name`` (of the ``code`` precision) on ``grid`` x ``block`` threads.

    Arguments follow kernels.cu's conventions: a DeviceArray is passed as its
    pointer, a float as a double (scalar settings are double, as in
    kernels.c), an int as a long. Kernels run in order on the GPU, while
    Python carries on: only copying values back to the host waits for them.
    """

    holders = []
    for arg in args:
        if isinstance(arg, DeviceArray):
            holders.append(ctypes.c_uint64(arg.pointer))
        elif isinstance(arg, float):
            holders.append(ctypes.c_double(arg))
        elif isinstance(arg, int):
            holders.append(ctypes.c_long(arg))
        else:
            raise TypeError(f"cannot pass {type(arg).__name__} to a kernel")
    pointers = (ctypes.c_void_p * len(holders))(*[ctypes.addressof(h) for h in holders])
    grid = grid if isinstance(grid, tuple) else (grid,)
    block = block if isinstance(block, tuple) else (block,)
    grid, block = (grid + (1, 1))[:3], (block + (1, 1))[:3]
    _check(_driver.cuLaunchKernel(_function(name, code), *grid, *block, shared, None, pointers, None), name)


def blocks(count: int, size: int = BLOCK) -> int:
    """Blocks of ``size`` threads that cover ``count`` values (at least 1)."""

    return max(1, (count + size - 1) // size)


# ---------------------------------------------------------------------------
# GPU memory
# ---------------------------------------------------------------------------

# Freed blocks are kept for reuse, by size, instead of handed back to the
# driver: cuMemFree waits for the GPU to finish, and cuMemAlloc is slow.
# Sizes round up (to 512 bytes, then to a quarter of the next power of two)
# so a block fits many requests. Reuse is safe without waiting: every kernel
# runs in order on one stream, so the next kernel to write the block runs
# after the last one that read it. (PyTorch's caching allocator, in small.)
_free_blocks: dict[int, list[int]] = {}
pool_bytes = 0                    # bytes allocated from the driver


def _rounded(size: int) -> int:
    if size <= 1 << 20:
        return max(512, (size + 511) // 512 * 512)
    step = 1 << (size.bit_length() - 3)
    return (size + step - 1) // step * step


def _allocate(size: int) -> int:
    global pool_bytes
    size = _rounded(size)
    kept = _free_blocks.get(size)
    if kept:
        return kept.pop()
    pointer = ctypes.c_uint64()
    result = _driver.cuMemAlloc_v2(ctypes.byref(pointer), size)
    if result == 2:                                   # out of memory: give the kept blocks back, retry
        release_memory()
        result = _driver.cuMemAlloc_v2(ctypes.byref(pointer), size)
    _check(result, f"cuMemAlloc({size})")
    pool_bytes += size
    return pointer.value


def _release(pointer: int, size: int) -> None:
    _free_blocks.setdefault(_rounded(size), []).append(pointer)


def release_memory() -> None:
    """Hand every kept (unused) block back to the driver."""

    global pool_bytes
    _check(_driver.cuCtxSynchronize(), "cuCtxSynchronize")
    for size, pointers in _free_blocks.items():
        for pointer in pointers:
            _driver.cuMemFree_v2(pointer)
            pool_bytes -= size
    _free_blocks.clear()


class DeviceArray:
    """``count`` numbers in GPU memory: float64 ("d"), float32 ("f") or int64 ("l", index tables).

    Behaves like a flat ``array`` for reading: Python access copies the
    values to the host once and keeps that copy (``_host``) until a kernel
    writes to the array (kernels that write in place call ``changed()``).
    """

    __slots__ = ("pointer", "count", "typecode", "_host", "__weakref__")

    def __init__(self, code: str, values: Sequence[float] = ()) -> None:
        """Like ``array(code, values)``: the values, uploaded."""

        host = values if type(values) is array and values.typecode == code else array(code, values)
        self._allocate(len(host), code)
        self._upload(host)

    def _allocate(self, count: int, code: str) -> None:
        self.count, self.typecode, self._host = count, code, None
        self.pointer = _allocate(max(1, count) * _ITEM_SIZE[code])

    @classmethod
    def empty(cls, count: int, code: str) -> DeviceArray:
        """``count`` numbers of GPU memory, not initialized."""

        out = cls.__new__(cls)
        out._allocate(count, code)
        return out

    def __del__(self) -> None:
        if _release is not None and hasattr(self, "pointer"):   # not at shutdown, nor if allocating failed
            _release(self.pointer, max(1, self.count) * _ITEM_SIZE[self.typecode])

    @property
    def nbytes(self) -> int:
        return self.count * _ITEM_SIZE[self.typecode]

    def changed(self) -> None:
        self._host = None

    def host(self) -> array:
        """The values as a host ``array`` (copied once, then kept; do not modify it)."""

        global syncs
        if self._host is None:
            values = array(self.typecode, bytes(self.nbytes))
            if self.count:
                address = values.buffer_info()[0]
                _check(_driver.cuMemcpyDtoH_v2(address, self.pointer, self.nbytes), "cuMemcpyDtoH")
            self._host = values
            syncs += 1
        return self._host

    # The sequence protocol, through the host copy.
    def __len__(self) -> int:
        return self.count

    def __iter__(self):
        return iter(self.host())

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self.count)
            if step == 1:
                return self.slice(start, max(start, stop))
            return array(self.typecode, self.host()[key])
        return self.host()[key]

    def __setitem__(self, key, value) -> None:
        if isinstance(key, slice) and isinstance(value, DeviceArray):
            start, stop, step = key.indices(self.count)
            if step == 1 and stop - start == value.count:        # a run from another DeviceArray: on the GPU
                _copy(_device(value, self.typecode), 0, self, start, value.count)
                return
            value = value.host()
        values = array(self.typecode, self.host())
        values[key] = value
        if len(values) != self.count:
            raise ValueError("cannot change a DeviceArray's length")
        self._upload(values)

    def __repr__(self) -> str:
        return f"DeviceArray({self.typecode!r}, {list(self.host())!r})"

    def __reduce__(self):                             # pickling saves the values
        return (_from_host, (self.typecode, self.host().tobytes()))

    def __eq__(self, other) -> bool:
        return list(self) == list(other)

    def __mul__(self, times: int) -> DeviceArray:     # repetition, as for arrays and lists
        out = DeviceArray.empty(self.count * max(0, times), self.typecode)
        if out.count and self.typecode in ("d", "f"):
            launch("repeat_values", self.typecode, blocks(out.count), BLOCK, self, self.count, out, out.count)
        else:
            for copy in range(max(0, times)):
                _copy(self, 0, out, copy * self.count, self.count)
        return out

    def __add__(self, other) -> DeviceArray:          # concatenation
        other = _device(other, self.typecode)
        out = DeviceArray.empty(self.count + other.count, self.typecode)
        _copy(self, 0, out, 0, self.count)
        _copy(other, 0, out, self.count, other.count)
        return out

    def slice(self, start: int, stop: int) -> DeviceArray:
        """Values [start, stop) as a new DeviceArray (copied on the GPU)."""

        out = DeviceArray.empty(stop - start, self.typecode)
        _copy(self, start, out, 0, stop - start)
        return out

    def _upload(self, values: array) -> None:
        if self.count:
            _check(_driver.cuMemcpyHtoD_v2(self.pointer, values.buffer_info()[0], self.nbytes), "cuMemcpyHtoD")
        self._host = None


def _copy(source: DeviceArray, start: int, target: DeviceArray, offset: int, count: int) -> None:
    if count:
        size = _ITEM_SIZE[source.typecode]
        _check(_driver.cuMemcpyDtoDAsync_v2(target.pointer + offset * size, source.pointer + start * size,
                                            count * size, None), "cuMemcpyDtoD")
        target.changed()


def _from_host(code: str, raw: bytes) -> DeviceArray:
    values = array(code)
    values.frombytes(raw)
    return _device(values, code)


def _device(values, code: str | None = None) -> DeviceArray:
    """``values`` as a DeviceArray of precision ``code`` (default: the current one).
    A DeviceArray of that precision passes through; anything else is uploaded."""

    code = code or typecode
    if isinstance(values, DeviceArray):
        if values.typecode == code:
            return values
        values = values.host()
    return DeviceArray(code, values)


def storage(values: Sequence[float]) -> DeviceArray:
    return _device(values)


def zeros(count: int, code: str | None = None) -> DeviceArray:
    out = DeviceArray.empty(count, code or typecode)
    if count:
        _check(_driver.cuMemsetD8Async(out.pointer, 0, out.nbytes, None), "cuMemsetD8")
    return out


def index_table(indices: Sequence[int]) -> DeviceArray:
    """A fixed table of offsets, uploaded once. Its host copy is kept: tables never change."""

    host = indices if type(indices) is array and indices.typecode == "l" else array("l", indices)
    out = _device(host, "l")
    out._host = host
    return out


def _indices(values: Sequence[int]) -> DeviceArray:
    """Integer arguments (targets, ids) on the GPU."""

    return values if isinstance(values, DeviceArray) and values.typecode == "l" else _device(values, "l")


def _like(x: DeviceArray) -> DeviceArray:
    """A new, uninitialized array of x's length and precision."""

    return DeviceArray.empty(x.count, x.typecode)


def synchronize() -> None:
    """Wait for every launched kernel to finish (for timing)."""

    _check(_driver.cuCtxSynchronize(), "cuCtxSynchronize")


# ---------------------------------------------------------------------------
# Elementwise kernels (kernels.cu, "Elementwise")
# ---------------------------------------------------------------------------


def _map(name: str, *inputs, scalars: tuple = ()) -> DeviceArray:
    """out[i] = f(inputs[...][i], scalars) with one thread per value."""

    first = _device(inputs[0])
    arrays = [first] + [_device(x, first.typecode) for x in inputs[1:]]
    if any(x.count != first.count for x in arrays):
        raise ValueError("elementwise kernels need equal lengths")
    out = _like(first)
    if first.count:
        launch(name, first.typecode, blocks(first.count), BLOCK, *arrays, *scalars, out, first.count)
    return out


def add(x, y) -> DeviceArray:
    return _map("add_vectors", x, y)


def multiply(x, y) -> DeviceArray:
    return _map("multiply_vectors", x, y)


def scale(x, factor: float) -> DeviceArray:
    return _map("scale_vector", x, scalars=(float(factor),))


def add_scaled(x, y, factor: float) -> DeviceArray:
    return _map("add_scaled", x, y, scalars=(float(factor),))


def reciprocal(x) -> DeviceArray:
    return _map("reciprocal_vector", x)


def exp(x) -> DeviceArray:
    return _map("exp_vector", x)


def log(x) -> DeviceArray:
    return _map("log_vector", x)


def sqrt(x) -> DeviceArray:
    return _map("sqrt_vector", x)


def relu(x) -> DeviceArray:
    return _map("relu_forward", x)


def relu_backward(x, grad) -> DeviceArray:
    return _map("relu_backward", x, grad)


def tanh(x) -> DeviceArray:
    return _map("tanh_forward", x)


def tanh_backward(y, grad) -> DeviceArray:
    return _map("tanh_backward", y, grad)


def sigmoid(x) -> DeviceArray:
    return _map("sigmoid_forward", x)


def sigmoid_backward(y, grad) -> DeviceArray:
    return _map("sigmoid_backward", y, grad)


def expand_repeated(x, outer: int, size: int, inner: int) -> DeviceArray:
    x = _device(x)
    if x.count != size:
        raise ValueError("expand_repeated needs size values")
    out = DeviceArray.empty(outer * size * inner, x.typecode)
    if out.count:
        launch("expand_repeated", x.typecode, blocks(out.count), BLOCK, x, out, size, inner, out.count)
    return out


# ---------------------------------------------------------------------------
# Matrix products (gemm) (kernels.cu, same section)
# ---------------------------------------------------------------------------

# Tile edges of kernels.cu's gemm kernels (gemm_128x128_nn, ...), biggest
# first. Float64 uses only the smallest: its arithmetic runs at 1/64 of
# float32's speed on this GPU (GeForce cards), so even 2 x 2 patches keep
# it busy, and small tiles give more blocks; the big tiles would not save
# anything (and an 8 x 8 patch of doubles does not fit in registers).
_GEMM_TILES = {"f": (128, 64, 32), "d": (32,)}
_GEMM_BK = {128: 8, 64: 16, 32: 16}           # each kernel's k step
_GEMM_THREADS = 256                           # threads per block, for every tile size
_GEMM_SLICE_STEPS = 8                         # split-k: at least this many k steps per slice
_gemm_multiprocessors = 0


def _multiprocessors() -> int:
    global _gemm_multiprocessors
    if not _gemm_multiprocessors:
        count = ctypes.c_int()
        _check(_driver.cuDeviceGetAttribute(ctypes.byref(count), 16, 0), "multiprocessor count")
        _gemm_multiprocessors = count.value
    return _gemm_multiprocessors


def gemm(a, b, batches: int, m: int, k: int, n: int, a_step: int, b_step: int, c_step: int,
         trans_a: bool, trans_b: bool, initial=None) -> DeviceArray:
    """Batched C = op(A) @ op(B) (+ initial), as kernels_c.gemm.

    ``initial`` is consumed: a DeviceArray of this precision is the result
    itself, added to in place.
    """

    a = _device(a)
    code = a.typecode
    b = _device(b, code)
    size = c_step * (batches - 1) + m * n if c_step else m * n
    if a.count < a_step * (batches - 1) + m * k or b.count < b_step * (batches - 1) + k * n:
        raise ValueError("gemm inputs are too short for these sizes")    # a GPU would read past them
    summed = c_step == 0 or batches == 1          # one output (all batch entries add into it)
    if not summed and c_step < m * n:
        # Batch entries whose outputs overlap must add one after another;
        # nothing uses this, so it runs on the C kernel.
        return _fallback("gemm")(a, b, batches, m, k, n, a_step, b_step, c_step, trans_a, trans_b, initial)
    if initial is None:
        gaps = not summed and c_step > m * n      # numbers between the outputs stay 0
        c = zeros(size, code) if gaps or k == 0 else DeviceArray.empty(size, code)
    else:
        c = _device(initial, code)
        if c.count != size:
            raise ValueError("gemm initial values have the wrong size")
    if m == 0 or n == 0 or k == 0 or batches == 0:
        return c
    accumulate = int(initial is not None)

    # A matrix-vector product with the matrix stored along k: a dot product
    # per output, one warp each (a single row or column of A or B is
    # stored the same way transposed or not).
    a_along_k, b_along_k = not trans_a or m == 1, trans_b or n == 1
    if (n == 1 and a_along_k) or (m == 1 and b_along_k):
        outputs = m if n == 1 else n
        a_row, b_row = (k, 0) if n == 1 else (0, k)
        warps = outputs if summed else batches * outputs
        launch("gemm_dots", code, blocks(32 * warps), BLOCK, a, b, c, batches, outputs, k,
               a_step, b_step, c_step, a_row, b_row, int(summed), accumulate)
        c.changed()
        return c

    # The biggest tile that still gives every multiprocessor a block, counting
    # the slices a single output's long sum (k, times the batch entries)
    # could be split into, and skipping tiles that would be mostly empty.
    processors = _multiprocessors()
    for edge in _GEMM_TILES[code]:
        tiles = -(-m // edge) * -(-n // edge)
        steps = batches * -(-k // _GEMM_BK[edge])
        reach = max(1, steps // _GEMM_SLICE_STEPS) if summed else batches
        if tiles * reach >= processors and 2 * m * n >= tiles * edge * edge:
            break
    grid = (-(-n // edge), -(-m // edge))
    name = f"gemm_{edge}x{edge}_{'t' if trans_a else 'n'}{'t' if trans_b else 'n'}"
    if not summed:
        launch(name, code, (*grid, min(batches, 65535)), _GEMM_THREADS, a, b, c, batches, m, k, n,
               a_step, b_step, c_step, accumulate, 0)
        c.changed()
        return c

    # Fewer tiles than multiprocessors: split the long sum into slices,
    # each written to its own partial result, added up afterwards (split-k).
    slices = 1 if tiles >= processors else min(-(-processors // tiles), max(1, steps // _GEMM_SLICE_STEPS), 64)
    if slices == 1:
        launch(name, code, grid, _GEMM_THREADS, a, b, c, batches, m, k, n, a_step, b_step, 0, accumulate, 1)
    else:
        share = -(-steps // slices)
        slices = -(-steps // share)                # no empty slices
        partials = DeviceArray.empty(slices * m * n, code)
        launch(name, code, (*grid, slices), _GEMM_THREADS, a, b, partials, batches, m, k, n,
               a_step, b_step, 0, 0, slices)
        launch("gemm_sum_slices", code, blocks(m * n), BLOCK, partials, c, m * n, slices, accumulate)
    c.changed()
    return c


# (end of gemm)


# ---------------------------------------------------------------------------
# Convolutions, pooling, gathers and scatters (kernels.cu, same section)
# ---------------------------------------------------------------------------

CONV_TILE, CONV_STEP, CONV_THREADS = 64, 16, 256   # as in kernels.cu
_CONV_BUSY = 256                  # blocks that keep the GPU busy (conv_kernel_grad splits its sum to reach it)


def _conv_inputs(first, *others):
    """The float inputs on the GPU, in the precision of the first."""

    first = _device(first)
    return (first,) + tuple(_device(x, first.typecode) for x in others)


import weakref                    # (for the cache of inverted index tables below)

_conv_inverses: dict[int, tuple] = {}   # id(table) -> (weak reference to it, {targets: (starts, sources)})


def _conv_inverse(index, targets: int) -> tuple[DeviceArray, DeviceArray] | None:
    """The index table inverted, for scatters done as gathers: entries
    sources[starts[t] : starts[t + 1]] (ascending) are those with index[e] == t.

    Built on the host, so only for tables whose values are on the host
    already (index_table keeps them: tables never change), and cached per
    table; None for indices that live only on the GPU (max-pool winners).
    """

    if isinstance(index, DeviceArray):
        if index._host is None:
            return None
        host = index._host
        cached = _conv_inverses.get(id(index))
        if cached is None or cached[0]() is not index:
            cached = (weakref.ref(index, lambda _, key=id(index): _conv_inverses.pop(key, None)), {})
            _conv_inverses[id(index)] = cached
        inverses = cached[1]
    else:
        host, inverses = index, {}
    if targets not in inverses:
        counts = [0] * (targets + 1)
        for t in host:
            counts[t + 1] += 1
        for t in range(targets):
            counts[t + 1] += counts[t]
        sources = sorted(range(len(host)), key=host.__getitem__)   # stable: ascending within a target
        inverses[targets] = (index_table(counts), index_table(sources))
    return inverses[targets]


def _conv_product_grid(rows: int, cols: int, depth: int) -> tuple[int, int, int]:
    """One block per 64 x 64 tile of a rows x cols product, for each of ``depth`` images."""

    return blocks(cols, CONV_TILE), blocks(rows, CONV_TILE), depth


def conv_forward(image, index, kernel, batches: int, image_size: int,
                 filters: int, patch_size: int, positions: int) -> DeviceArray:
    image, kernel = _conv_inputs(image, kernel)
    index = _indices(index)
    out = DeviceArray.empty(batches * filters * positions, image.typecode)
    if out.count:
        launch("conv_forward", image.typecode, _conv_product_grid(filters, positions, batches), CONV_THREADS,
               image, index, kernel, out, batches, image_size, filters, patch_size, positions)
    return out


def conv_kernel_grad(image, index, grad, batches: int, image_size: int,
                     filters: int, patch_size: int, positions: int) -> DeviceArray:
    image, grad = _conv_inputs(image, grad)
    index = _indices(index)
    code, count = image.typecode, filters * patch_size
    dkernel = zeros(count, code)
    if not count or not batches * positions:
        return dkernel
    # Split the sum over (image, position) into enough slices to fill the GPU:
    # runs of whole images when there are many, else parts of each image.
    grid = _conv_product_grid(filters, patch_size, 1)
    wanted = max(1, _CONV_BUSY // (grid[0] * grid[1]))
    if wanted <= batches:
        images_per_slice, slices_per_image, chunk = -(-batches // wanted), 1, positions
        slices = -(-batches // images_per_slice)
    else:
        parts = min(-(-wanted // batches), blocks(positions, CONV_STEP))
        chunk = blocks(-(-positions // parts), CONV_STEP) * CONV_STEP
        images_per_slice, slices_per_image = 1, -(-positions // chunk)
        slices = batches * slices_per_image
    partial = dkernel if slices == 1 else DeviceArray.empty(slices * count, code)
    launch("conv_kernel_grad", code, grid[:2] + (slices,), CONV_THREADS, image, index, grad, partial,
           batches, image_size, filters, patch_size, positions, images_per_slice, slices_per_image, chunk)
    if slices > 1:
        launch("conv_sum_slices", code, blocks(count), BLOCK, partial, dkernel, slices, count)
    return dkernel


def conv_input_grad(grad, kernel, index, batches: int, image_size: int,
                    filters: int, patch_size: int, positions: int) -> DeviceArray:
    grad, kernel = _conv_inputs(grad, kernel)
    index = _indices(index)
    image_grad = zeros(batches * image_size, grad.typecode)
    if image_grad.count and positions and patch_size:
        launch("conv_input_grad", grad.typecode, _conv_product_grid(positions, patch_size, batches), CONV_THREADS,
               grad, kernel, index, image_grad, batches, image_size, filters, patch_size, positions)
    return image_grad


def conv_direct(image, kernel, batches: int, channels: int, height: int, width: int,
                filters: int, kernel_rows: int, kernel_cols: int) -> DeviceArray:
    image, kernel = _conv_inputs(image, kernel)
    out = DeviceArray.empty(batches * filters * (height - kernel_rows + 1) * (width - kernel_cols + 1), image.typecode)
    if out.count:
        launch("conv_direct", image.typecode, blocks(out.count), BLOCK, image, kernel, out, batches,
               channels, height, width, filters, kernel_rows, kernel_cols)
    return out


def pad_planes(x, planes: int, rows: int, cols: int, pad_rows: int, pad_cols: int) -> DeviceArray:
    x = _device(x)
    out = DeviceArray.empty(planes * (rows + 2 * pad_rows) * (cols + 2 * pad_cols), x.typecode)
    if out.count:
        launch("pad_planes", x.typecode, blocks(out.count), BLOCK, x, out, planes, rows, cols, pad_rows, pad_cols)
    return out


def crop_planes(padded, planes: int, rows: int, cols: int, pad_rows: int, pad_cols: int) -> DeviceArray:
    padded = _device(padded)
    out = DeviceArray.empty(planes * rows * cols, padded.typecode)
    if out.count:
        launch("crop_planes", padded.typecode, blocks(out.count), BLOCK, padded, out,
               planes, rows, cols, pad_rows, pad_cols)
    return out


def gather_indexed(image, index, batches: int, image_size: int) -> DeviceArray:
    image, index = _device(image), _indices(index)
    out = DeviceArray.empty(batches * index.count, image.typecode)
    if out.count:
        launch("gather_indexed", image.typecode, blocks(out.count), BLOCK, image, index, out,
               batches, image_size, index.count)
    return out


def scatter_add_indexed(grad, index, batches: int, image_size: int, into: DeviceArray | None = None) -> DeviceArray:
    if into is not None and (not isinstance(into, DeviceArray) or into.typecode != typecode):
        raise TypeError("scatter_add_indexed(into=...) needs a buffer from zeros() in the current precision")
    grad = _device(grad, into.typecode if into is not None else None)
    image_grad = zeros(batches * image_size, grad.typecode) if into is None else into
    count = len(index)
    if not batches * count:
        return image_grad
    inverse = _conv_inverse(index, image_size)
    if inverse is not None:                       # a gather: reproducible
        launch("scatter_add_indexed", grad.typecode, blocks(batches * image_size), BLOCK, grad, *inverse,
               image_grad, batches, image_size, count)
    else:                                         # atomics: the indices are only on the GPU
        launch("scatter_add_indexed_atomic", grad.typecode, blocks(batches * count), BLOCK, grad, _indices(index),
               image_grad, batches, image_size, count)
    image_grad.changed()
    return image_grad


def depthwise_forward(image, index, kernel, batches: int, channels: int, channel_size: int,
                      q_size: int) -> DeviceArray:
    image, kernel = _conv_inputs(image, kernel)
    index = _indices(index)
    positions = index.count // q_size
    out = DeviceArray.empty(batches * channels * positions, image.typecode)
    if out.count:
        launch("depthwise_forward", image.typecode, blocks(out.count), BLOCK, image, index, kernel, out,
               batches, channels, channel_size, positions, q_size)
    return out


def depthwise_backward(image, index, kernel, grad, batches: int, channels: int, channel_size: int,
                       q_size: int) -> tuple[DeviceArray, DeviceArray]:
    image, kernel, grad = _conv_inputs(image, kernel, grad)
    inverse = _conv_inverse(index, channel_size) or _conv_inverse(list(index), channel_size)
    index = _indices(index)
    code, positions = image.typecode, index.count // q_size
    dkernel = DeviceArray.empty(channels * q_size, code)
    dimage = DeviceArray.empty(batches * channels * channel_size, code)
    sizes = (batches, channels, channel_size, positions, q_size)
    if dkernel.count:
        launch("depthwise_kernel_grad", code, dkernel.count, CONV_THREADS, image, index, grad, dkernel, *sizes)
    if dimage.count:
        launch("depthwise_image_grad", code, blocks(dimage.count), BLOCK, *inverse, kernel, grad, dimage, *sizes)
    return dkernel, dimage


def csr_matmul(row_start, cols, values, dense, rows: int, n: int) -> DeviceArray:
    values, dense = _conv_inputs(values, dense)
    row_start, cols = _indices(row_start), _indices(cols)
    out = DeviceArray.empty(rows * n, values.typecode)
    if out.count:
        launch("csr_matmul", values.typecode, blocks(out.count), BLOCK, row_start, cols, values, dense, out, rows, n)
    return out


def gather_rows(table, ids, width: int) -> DeviceArray:
    table, ids = _device(table), _indices(ids)
    out = DeviceArray.empty(ids.count * width, table.typecode)
    if out.count:
        launch("gather_rows", table.typecode, blocks(out.count), BLOCK, table, ids, out, ids.count, width)
    return out


def scatter_add_rows(grad, ids, width: int, rows: int, into: DeviceArray | None = None) -> DeviceArray:
    if into is not None and (not isinstance(into, DeviceArray) or into.typecode != typecode
                             or into.count != rows * width):
        raise TypeError("scatter_add_rows(into=...) needs a table in the current precision")
    grad, ids = _device(grad, into.typecode if into is not None else None), _indices(ids)
    table_grad = zeros(rows * width, grad.typecode) if into is None else into
    if ids.count * width:
        launch("scatter_add_rows", grad.typecode, blocks(ids.count * width), BLOCK, grad, ids, table_grad,
               ids.count, width)
        table_grad.changed()
    return table_grad


def window_mean(image, index, planes: int, plane_size: int, window_size: int) -> DeviceArray:
    image, index = _device(image), _indices(index)
    windows = index.count // window_size
    out = DeviceArray.empty(planes * windows, image.typecode)
    if out.count:
        launch("window_mean", image.typecode, blocks(out.count), BLOCK, image, index, out,
               planes, plane_size, windows, window_size)
    return out


def window_mean_backward(grad, index, planes: int, plane_size: int, window_size: int) -> DeviceArray:
    grad = _device(grad)
    inverse = _conv_inverse(index, plane_size) or _conv_inverse(list(index), plane_size)
    windows = len(index) // window_size
    image_grad = DeviceArray.empty(planes * plane_size, grad.typecode)
    if image_grad.count:
        launch("window_mean_backward", grad.typecode, blocks(image_grad.count), BLOCK, grad, *inverse, image_grad,
               planes, plane_size, windows, window_size)
    return image_grad


def window_max(image, index, planes: int, plane_size: int, window_size: int) -> tuple[DeviceArray, DeviceArray]:
    """The maxima and, as int64 offsets on the GPU, where they were (max_pool2d's
    backward pass hands them to scatter_add_indexed without a copy)."""

    image, index = _device(image), _indices(index)
    windows = index.count // window_size
    out = DeviceArray.empty(planes * windows, image.typecode)
    argmax = DeviceArray.empty(planes * windows, "l")
    if out.count:
        launch("window_max", image.typecode, blocks(out.count), BLOCK, image, index, out, argmax,
               planes, plane_size, windows, window_size)
    return out, argmax


def sum_repeated(x, outer: int, size: int, inner: int) -> DeviceArray:
    if len(x) != outer * size * inner:
        raise ValueError("sum_repeated needs outer * size * inner values")
    x = _device(x)
    out = DeviceArray.empty(size, x.typecode)
    if inner == 1 and size >= BLOCK:              # many sums: one thread each
        launch("sum_repeated_rows", x.typecode, blocks(size), BLOCK, x, out, outer, size)
    elif size:                                    # few, long sums: one block each
        launch("sum_repeated", x.typecode, size, CONV_THREADS, x, out, outer, size, inner)
    return out

# (end of conv)


# ---------------------------------------------------------------------------
# Rows and channels: softmax, norms, losses, GELU, the optimizer (kernels.cu, same section)
# ---------------------------------------------------------------------------

_REDUCE_BLOCKS = 256              # blocks for a sum over a whole array (sum_of_squares, bce_with_logits_sum)


_WARP_ROWS = 8                    # rows per block when a warp takes each row
_WIDEST_WARP_ROW = 1024           # rows wider than this get a block of BLOCK threads each


def _launch_rows(name: str, code: str, rows: int, width: int, *args) -> None:
    """Launch row kernel ``name`` with a team of threads per row (kernels.cu, idea 1):
    a warp per row (8 rows per 32 x 8 block) or, for wide rows, a block per row."""

    if not rows:
        return
    if width <= _WIDEST_WARP_ROW:
        launch(name, code, blocks(rows, _WARP_ROWS), (32, _WARP_ROWS), *args)
    else:
        launch(name, code, rows, BLOCK, *args)


def _rows(name: str, x, width: int, *arrays) -> DeviceArray:
    """y = a row kernel over x and ``arrays``, with a new output of x's shape."""

    x = _device(x)
    arrays = [_device(a, x.typecode) for a in arrays]
    y, rows = _like(x), x.count // width
    _launch_rows(name, x.typecode, rows, width, x, *arrays, y, rows, width)
    return y


def _add_column_sums(out: DeviceArray, a: DeviceArray, rows: int, width: int, b=None, scale=None) -> None:
    """out[s] += the sum over rows r of a[r, s] * b[r, s] * scale[r] (b and scale optional),
    in two deterministic passes (kernels.cu, idea 2)."""

    if not rows:
        return
    chunk = max(16, -(-rows // 1024))              # rows per partial sum; at most 1024 partials
    chunks = -(-rows // chunk)
    partial = DeviceArray.empty(chunks * width, "d")
    code = a.typecode
    launch("rows_column_partials", code, (blocks(width), chunks), BLOCK, a,
           a if b is None else b, a if scale is None else scale, int(b is not None), int(scale is not None),
           partial, rows, width, chunk)
    launch("rows_column_finish", code, blocks(width), BLOCK, partial, out, chunks, width)
    out.changed()


def _total(name: str, x: DeviceArray, *arrays) -> float:
    """A sum over the whole array: per-block partial sums, added up by one block, read back as a float."""

    if not x.count:
        return 0.0
    count = min(_REDUCE_BLOCKS, blocks(x.count))
    partial, total = DeviceArray.empty(count, "d"), DeviceArray.empty(1, "d")
    launch(name, x.typecode, count, BLOCK, x, *arrays, partial, x.count)
    launch("rows_sum_partials", x.typecode, 1, BLOCK, partial, total, count)
    return total.host()[0]


def gelu(x) -> tuple[DeviceArray, DeviceArray]:
    x = _device(x)
    y, cdf = _like(x), _like(x)
    if x.count:
        launch("gelu_forward", x.typecode, blocks(x.count), BLOCK, x, y, cdf, x.count)
    return y, cdf


def gelu_backward(x, cdf, grad) -> DeviceArray:
    return _map("gelu_backward", x, cdf, grad)


def softmax_rows(x, width: int) -> DeviceArray:
    return _rows("softmax_rows", x, width)


def softmax_rows_backward(y, grad, width: int) -> DeviceArray:
    return _rows("softmax_rows_backward", y, width, grad)


def softmax_xent_rows(x, targets, factor: float, width: int) -> tuple[DeviceArray, DeviceArray]:
    """In place: each row of logits x becomes factor * (softmax - one_hot(target)); also the rows' losses (float64)."""

    if not isinstance(x, DeviceArray) or x.typecode == "l":
        raise TypeError("softmax_xent_rows works in place on a DeviceArray")
    targets = _indices(targets)
    nll = DeviceArray.empty(targets.count, "d")
    _launch_rows("softmax_xent_rows", x.typecode, targets.count, width,
                 x, targets, float(factor), nll, targets.count, width)
    x.changed()
    return x, nll


def add_rows(x, out, width: int) -> DeviceArray:
    """In place: out[s] += every row's x[., s]."""

    if not isinstance(out, DeviceArray) or out.typecode == "l" or out.count != width:
        raise TypeError("add_rows adds into a DeviceArray one row wide")
    x = _device(x, out.typecode)
    _add_column_sums(out, x, x.count // width, width)
    return out


def layer_norm_rows(x, weight, bias, width: int, eps: float) -> tuple[DeviceArray, DeviceArray, DeviceArray]:
    x = _device(x)
    weight, bias = _device(weight, x.typecode), _device(bias, x.typecode)
    rows = x.count // width
    y, x_hat, inv_std = _like(x), _like(x), DeviceArray.empty(rows, x.typecode)
    _launch_rows("layer_norm_rows", x.typecode, rows, width,
                 x, weight, bias, y, x_hat, inv_std, rows, width, float(eps))
    return y, x_hat, inv_std


def layer_norm_rows_backward(grad, x_hat, inv_std, weight, width: int, need_dx: bool):
    grad = _device(grad)
    x_hat, inv_std, weight = (_device(a, grad.typecode) for a in (x_hat, inv_std, weight))
    rows = inv_std.count
    dx = None
    if need_dx:
        dx = _like(grad)
        _launch_rows("layer_norm_rows_backward", grad.typecode, rows, width,
                     grad, x_hat, inv_std, weight, dx, rows, width)
    dweight, dbias = zeros(width, grad.typecode), zeros(width, grad.typecode)
    _add_column_sums(dweight, grad, rows, width, b=x_hat)
    _add_column_sums(dbias, grad, rows, width)
    return dx, dweight, dbias


def rms_norm_rows(x, weight, width: int, eps: float) -> tuple[DeviceArray, DeviceArray]:
    x = _device(x)
    weight = _device(weight, x.typecode)
    rows = x.count // width
    y, inv_rms = _like(x), DeviceArray.empty(rows, x.typecode)
    _launch_rows("rms_norm_rows", x.typecode, rows, width, x, weight, y, inv_rms, rows, width, float(eps))
    return y, inv_rms


def rms_norm_rows_backward(grad, x, inv_rms, weight, width: int) -> tuple[DeviceArray, DeviceArray]:
    grad = _device(grad)
    x, inv_rms, weight = (_device(a, grad.typecode) for a in (x, inv_rms, weight))
    rows = inv_rms.count
    dx, dweight = _like(grad), zeros(width, grad.typecode)
    _launch_rows("rms_norm_rows_backward", grad.typecode, rows, width, grad, x, inv_rms, weight, dx, rows, width)
    _add_column_sums(dweight, grad, rows, width, b=x, scale=inv_rms)
    return dx, dweight


def batch_norm_channels(x, weight, bias, batch: int, channels: int, plane: int, eps: float):
    x = _device(x)
    weight, bias = _device(weight, x.typecode), _device(bias, x.typecode)
    y, x_hat = _like(x), _like(x)
    inv_std, mean, variance = (DeviceArray.empty(channels, x.typecode) for _ in range(3))
    if channels and batch * plane:
        launch("batch_norm_channels", x.typecode, channels, 2 * BLOCK, x, weight, bias,
               y, x_hat, inv_std, mean, variance, batch, channels, plane, float(eps))
    return y, x_hat, inv_std, mean, variance


def batch_norm_channels_backward(grad, x_hat, inv_std, weight, batch: int, channels: int, plane: int):
    grad = _device(grad)
    x_hat, inv_std, weight = (_device(a, grad.typecode) for a in (x_hat, inv_std, weight))
    dx, dweight, dbias = _like(grad), zeros(channels, grad.typecode), zeros(channels, grad.typecode)
    if channels and batch * plane:
        launch("batch_norm_channels_backward", grad.typecode, channels, 2 * BLOCK,
               grad, x_hat, inv_std, weight, dx, dweight, dbias, batch, channels, plane)
    return dx, dweight, dbias


def sum_of_squares(x) -> float:
    return _total("sum_of_squares", _device(x))


def bce_with_logits_sum(x, t) -> float:
    x = _device(x)
    t = _device(t, x.typecode)
    if x.count != t.count:
        raise ValueError("bce_with_logits_sum needs equal lengths")
    return _total("bce_with_logits_sum", x, t)


def bce_with_logits_backward(x, t, g: float) -> DeviceArray:
    return _map("bce_with_logits_backward", x, t, scalars=(float(g),))


def dropout_mask(count: int, p: float, seed: int) -> DeviceArray:
    mask = DeviceArray.empty(count, typecode)
    seed &= (1 << 64) - 1
    if count:                                      # the seed's 64 bits, passed as a (signed) long
        launch("dropout_mask", typecode, blocks(count), BLOCK, mask, count, float(p),
               seed - (1 << 64) if seed >= 1 << 63 else seed)
    return mask


def projected_sign_step(delta, grad, x, alpha: float, eps: float) -> DeviceArray:
    return _map("projected_sign_step", delta, grad, x, scalars=(float(alpha), float(eps)))


def adam_update(w, grad, m, v, *hyperparameters) -> tuple[DeviceArray, DeviceArray, DeviceArray]:
    # As in kernels_c: m and v are the optimizer's own state, updated in place
    # once they are DeviceArrays of w's precision (the first call uploads them).
    w = _device(w)
    grad, m, v = (_device(a, w.typecode) for a in (grad, m, v))
    if not w.count == grad.count == m.count == v.count:
        raise ValueError("adam_update needs weights, gradient and moments of equal length")
    new_w = _like(w)
    if w.count:
        launch("adam_update", w.typecode, blocks(w.count), BLOCK, w, grad, m, v, new_w, w.count,
               *(float(h) for h in hyperparameters))
        m.changed()
        v.changed()
    return new_w, m, v

# (end of rows)


# ---------------------------------------------------------------------------
# Recurrent: the LSTM cell and layer, the CRF (kernels.cu, same section)
# ---------------------------------------------------------------------------

_LSTM_TILE = 16                   # LSTM_TILE in kernels.cu: the tiled products' blocks are 16 x 16 threads


def lstm_memory(gates, c, batch: int, hidden: int) -> DeviceArray:
    gates = _device(gates)
    c, out = _device(c, gates.typecode), DeviceArray.empty(batch * hidden, gates.typecode)
    if batch * hidden:
        launch("lstm_memory", gates.typecode, blocks(batch * hidden), BLOCK, gates, c, out, batch, hidden)
    return out


def lstm_output(gates, c_new, batch: int, hidden: int) -> DeviceArray:
    gates = _device(gates)
    c_new, out = _device(c_new, gates.typecode), DeviceArray.empty(batch * hidden, gates.typecode)
    if batch * hidden:
        launch("lstm_output", gates.typecode, blocks(batch * hidden), BLOCK, gates, c_new, out, batch, hidden)
    return out


def _lstm_cell_backward(name: str, gates, state, grad, batch: int, hidden: int) -> tuple[DeviceArray, DeviceArray]:
    gates = _device(gates)
    state, grad = _device(state, gates.typecode), _device(grad, gates.typecode)
    dgates = DeviceArray.empty(batch * 4 * hidden, gates.typecode)     # every value is written
    dstate = DeviceArray.empty(batch * hidden, gates.typecode)
    if batch * hidden:
        launch(name, gates.typecode, blocks(batch * hidden), BLOCK, gates, state, grad, dgates, dstate, batch, hidden)
    return dgates, dstate


def lstm_memory_backward(gates, c, grad, batch: int, hidden: int) -> tuple[DeviceArray, DeviceArray]:
    return _lstm_cell_backward("lstm_memory_backward", gates, c, grad, batch, hidden)


def lstm_output_backward(gates, c_new, grad, batch: int, hidden: int) -> tuple[DeviceArray, DeviceArray]:
    return _lstm_cell_backward("lstm_output_backward", gates, c_new, grad, batch, hidden)


def _lstm_tiles(rows: int, columns: int) -> tuple[int, int]:
    """The grid for a tiled product with a rows x columns output (x runs along columns)."""

    return blocks(columns, _LSTM_TILE), blocks(rows, _LSTM_TILE)


def lstm_sequence(input_gates, weight, h0, c0, batch: int, steps: int, hidden: int) -> tuple[DeviceArray, tuple]:
    """(packed, saved), as kernels_c's: two launches per time step (see kernels.cu)."""

    input_gates = _device(input_gates)
    code, state, width = input_gates.typecode, batch * hidden, 4 * hidden
    weight, h0, c0 = (_device(x, code) for x in (weight, h0, c0))
    gates = DeviceArray.empty(steps * batch * width, code)
    cs, hs = DeviceArray.empty((steps + 1) * state, code), DeviceArray.empty((steps + 1) * state, code)
    packed = DeviceArray.empty(batch * (steps + 1) * hidden, code)
    _copy(h0, 0, hs, 0, state)
    _copy(c0, 0, cs, 0, state)
    if state:
        for t in range(steps):
            launch("lstm_step_gates", code, _lstm_tiles(batch, width), (_LSTM_TILE, _LSTM_TILE),
                   input_gates, weight, hs, gates, t, batch, steps, hidden)
            launch("lstm_step_cell", code, blocks(state), BLOCK, gates, cs, hs, t, batch, hidden)
        launch("lstm_pack", code, blocks(packed.count), BLOCK, cs, hs, packed, batch, steps, hidden)
    return packed, (gates, cs, hs)


def lstm_sequence_backward(weight, saved: tuple, grad, batch: int, steps: int, hidden: int) -> tuple[DeviceArray, ...]:
    """(d input_gates, d weight, d h0, d c0): two launches per time step, then the weight's product."""

    gates, cs, hs = saved
    code, state, width = gates.typecode, batch * hidden, 4 * hidden
    weight, grad = _device(weight, code), _device(grad, code)
    dinput = DeviceArray.empty(batch * steps * width, code)            # every value is written
    dweight = zeros(width * hidden, code)
    dh, dh0, dc0 = DeviceArray.empty(state, code), DeviceArray.empty(state, code), DeviceArray.empty(state, code)
    if not state:
        return dinput, dweight, dh0, dc0
    launch("lstm_grad_start", code, blocks(state), BLOCK, grad, dh, dc0, batch, steps, hidden)   # dc0 is the carry
    for t in range(steps - 1, -1, -1):
        launch("lstm_step_cell_backward", code, blocks(state), BLOCK, gates, cs, dh, dc0, dinput, t, batch, steps, hidden)
        launch("lstm_step_recurrent", code, _lstm_tiles(batch, hidden), (_LSTM_TILE, _LSTM_TILE),
               dinput, weight, grad, dh if t > 0 else dh0, t, batch, steps, hidden)
    launch("lstm_weight_grad", code, _lstm_tiles(width, hidden), (_LSTM_TILE, _LSTM_TILE),
           dinput, hs, dweight, batch, steps, hidden)
    return dinput, dweight, dh0, dc0


def _crf_block(tags: int) -> int:
    """Threads for one sentence's block: one per tag, in whole warps (at most BLOCK; more tags loop)."""

    return min(BLOCK, (tags + 31) // 32 * 32)


def crf_forward(e, start, weights, batch: int, steps: int, tags: int) -> tuple[DeviceArray, tuple]:
    """(alpha, saved), as kernels_c's: one block per sentence, all words in one launch."""

    e = _device(e)
    code, n = e.typecode, batch * tags
    start, weights = _device(start, code), _device(weights, code)
    ps, qs = DeviceArray.empty((steps - 1) * n, code), DeviceArray.empty((steps - 1) * n, code)
    alpha = DeviceArray.empty(n, code)
    if n:
        launch("crf_forward", code, batch, _crf_block(tags), e, start, weights, ps, qs, alpha, batch, steps, tags,
               shared=2 * tags * _ITEM_SIZE[code])
    return alpha, (ps, qs)


def crf_forward_backward(weights, saved: tuple, grad, batch: int, steps: int, tags: int) -> tuple[DeviceArray, ...]:
    """(d e, d start, d weights)."""

    ps, qs = saved
    code, n = ps.typecode, batch * tags
    weights, grad = _device(weights, code), _device(grad, code)
    de, dstart, dweights = DeviceArray.empty(batch * steps * tags, code), zeros(tags, code), zeros(tags * tags, code)
    if not n:
        return de, dstart, dweights
    dqs = DeviceArray.empty((steps - 1) * n, code)
    launch("crf_forward_backward", code, batch, _crf_block(tags), weights, ps, qs, grad, de, dqs, batch, steps, tags,
           shared=2 * tags * _ITEM_SIZE[code])
    launch("crf_weights_grad", code, blocks(tags * tags), BLOCK, ps, dqs, dweights, (steps - 1) * batch, tags)
    launch("crf_start_grad", code, blocks(tags), BLOCK, de, dstart, batch, steps, tags)
    return de, dstart, dweights

# (end of recurrent)


# ---------------------------------------------------------------------------
# Kernels without a CUDA version yet: the C kernel on host copies
# ---------------------------------------------------------------------------

# Every kernel kernels.py dispatches to; each one not defined above gets a fallback.
KERNELS = (
    "gemm", "gelu", "gelu_backward", "softmax_rows", "softmax_xent_rows", "add_rows", "softmax_rows_backward",
    "layer_norm_rows", "layer_norm_rows_backward", "rms_norm_rows", "rms_norm_rows_backward",
    "batch_norm_channels", "batch_norm_channels_backward", "conv_forward", "conv_kernel_grad", "pad_planes",
    "crop_planes", "conv_direct", "conv_input_grad", "gather_indexed", "depthwise_forward", "depthwise_backward",
    "csr_matmul", "gather_rows", "scatter_add_rows", "scatter_add_indexed", "add", "multiply", "scale",
    "reciprocal", "exp", "log", "sqrt", "dropout_mask", "add_scaled", "projected_sign_step", "sum_of_squares",
    "adam_update", "sum_repeated", "sigmoid", "sigmoid_backward", "lstm_memory", "lstm_memory_backward",
    "lstm_output", "lstm_output_backward", "lstm_sequence", "lstm_sequence_backward", "crf_forward",
    "crf_forward_backward", "bce_with_logits_sum", "bce_with_logits_backward", "relu", "relu_backward", "tanh",
    "tanh_backward", "window_mean", "window_mean_backward", "window_max", "expand_repeated",
)
_LSTM_STEPS = {"lstm_memory", "lstm_output"}          # kernels_c names these through lstm_step(name, ...)
_fallbacks: set[str] = set()


def _to_host(value, originals: dict):
    if isinstance(value, DeviceArray):
        host = array(value.typecode, value.host())     # a private copy: the C kernel may write to it
        originals[id(host)] = value
        return host
    if isinstance(value, tuple):
        return tuple(_to_host(v, originals) for v in value)
    return value


def _to_device(value, originals: dict):
    if type(value) is array and value.typecode in ("d", "f"):
        original = originals.get(id(value))
        if original is not None:                     # an argument the C kernel changed in place
            original._upload(value)
            return original
        return _device(value, value.typecode)
    if isinstance(value, tuple):
        return tuple(_to_device(v, originals) for v in value)
    return value


def _fallback(name: str):
    def run(*args, **kwargs):
        originals = {}                               # id of each host copy -> its DeviceArray
        host_args = [_to_host(arg, originals) for arg in args]
        host_kwargs = {key: _to_host(arg, originals) for key, arg in kwargs.items()}
        previous = kernels_c.typecode
        kernels_c.set_precision(typecode)
        try:
            if name in _LSTM_STEPS or name.removesuffix("_backward") in _LSTM_STEPS:
                step = kernels_c.lstm_step_backward if name.endswith("_backward") else kernels_c.lstm_step
                result = step(name, *host_args, **host_kwargs)
            else:
                result = getattr(kernels_c, name)(*host_args, **host_kwargs)
        finally:
            kernels_c.set_precision(previous)
        return _to_device(result, originals)

    run.__name__ = name
    run.__doc__ = f"{name} on the C kernels, through host copies (no CUDA version yet)."
    return run


for _name in KERNELS:
    if _name not in globals():
        globals()[_name] = _fallback(_name)
        _fallbacks.add(_name)


def missing() -> list[str]:
    """The kernels that still run on the C fallback."""

    return sorted(_fallbacks)
