"""Small tensor container, shape-checked math helpers, and tensor autograd.

Tensor data is a flat sequence of floats in row-major order: a C array of
doubles when the C kernels are on, a plain list otherwise (see
``kernels.storage``). Ops index, slice and iterate it the same way either
way. The file reads top to bottom as:

1. ``Tensor``: the container and the backward pass
2. Recording ops: ``no_grad`` and ``_op_result``, the helper every op uses
3. Elementwise math with broadcasting
4. Matrix multiplication
5. Shape ops: reshape, permute, stack, concat, row gathers
6. Reductions: sum, mean, softmax
7. Convolution and pooling
8. Small index helpers

Performance notes, kept deliberately simple:

- Broadcasting and permutes precompute one flat "index map" per shape pair
  (cached), so the per-element work is a single list lookup.
- Every matrix product, forward and backward (``dA = dC @ B^T``,
  ``dB = A^T @ dC``), is one call to ``kernels.gemm``. That runs compiled C
  when a C compiler is available and plain-Python dot products otherwise
  (see kernels.py); both give the same numbers.
- Convolution is "im2col": gather every image patch into a row, then one
  batched ``kernels.gemm`` against the flattened kernels.
"""

from __future__ import annotations

import math
import operator
import random
from time import perf_counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from collections import OrderedDict
from functools import lru_cache, wraps
from typing import Callable, NamedTuple

import kernels


Number = int | float


# ---------------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------------


class Tensor:
    """A dense tensor backed by flat row-major data.

    ``data`` (and ``grad``) hold the values in the kernel backend's own form
    (see ``kernels.storage``): a C array of doubles when the C kernels are
    on, a list otherwise. Assigning any sequence of numbers converts it.
    """

    def __init__(
        self,
        data: Sequence[Number],
        shape: tuple[int, ...],
        *,
        requires_grad: bool = False,
        _children: tuple["Tensor", ...] = (),
        _op: str = "",
    ) -> None:
        if len(shape) == 0:
            raise ValueError("Tensor shape must have at least one dimension")
        if any(dim <= 0 for dim in shape):
            raise ValueError("shape dimensions must be positive")
        if len(data) != _numel(shape):
            raise ValueError("data length does not match shape")

        self.data = list(map(float, data))   # always a copy
        self.shape = tuple(shape)
        self.requires_grad = requires_grad
        self.grad = kernels.zeros(len(self.data)) if requires_grad else None

        self._prev = set(_children)
        self._op = _op
        self._backward: Callable[[], None] = _no_backward

    @property
    def data(self) -> Sequence[float]:
        return self._data

    @data.setter
    def data(self, values: Sequence[float]) -> None:
        self._data = kernels.storage(values)

    @property
    def grad(self) -> Sequence[float] | None:
        return self._grad

    @grad.setter
    def grad(self, values: Sequence[float] | None) -> None:
        if values is None or isinstance(values, RowGrad):
            self._grad = values
        else:
            self._grad = kernels.storage(values)

    sparse_grad = False   # True on an embedding table: its gradient is a RowGrad (see below)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, data={self.tolist()})"

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return len(self.data)

    def __getitem__(self, key) -> "float | Tensor":
        """``x[i, j, k]`` with one integer per axis is that number (a float).

        Anything else (fewer integers, slices like ``1:3`` or ``::2``, ``...``)
        selects a sub-tensor and returns it as a new Tensor that gradients
        flow back through; see ``take_slices``.
        """

        key = _integer_keys(key)
        if isinstance(key, int) and self.ndim == 1:
            return self.data[_normalize_index(key, self.shape[0])]
        if isinstance(key, tuple) and len(key) == self.ndim and all(isinstance(k, int) for k in key):
            return self.data[_flat_index(key, self.shape)]
        return take_slices(self, key)

    def item(self) -> float:
        if len(self.data) != 1:
            raise ValueError("item() needs a tensor with exactly one value")
        return self.data[0]

    def tolist(self) -> list:
        return _nested_list(self.data, self.shape)

    def detach(self) -> "Tensor":
        """Same values, cut off from the autograd graph."""

        return Tensor(self.data, self.shape)

    @classmethod
    def zeros(cls, shape: tuple[int, ...], *, requires_grad: bool = False) -> "Tensor":
        return cls([0.0] * _numel(shape), shape, requires_grad=requires_grad)

    @classmethod
    def from_list(
        cls,
        values: Sequence,
        *,
        requires_grad: bool = False,
    ) -> "Tensor":
        shape = _infer_shape(values)
        return cls(_flatten_nested(values), shape, requires_grad=requires_grad)

    @classmethod
    def stack(cls, tensors: Sequence["Tensor"], axis: int = 0) -> "Tensor":
        return stack(tensors, axis=axis)

    def __add__(self, other: Number | "Tensor") -> "Tensor":
        return add(self, other)

    def __radd__(self, other: Number | "Tensor") -> "Tensor":
        return add(self, other)

    def __sub__(self, other: Number | "Tensor") -> "Tensor":
        if isinstance(other, (int, float)):
            return add(self, -other)
        return subtract(self, other)

    def __rsub__(self, other: Number | "Tensor") -> "Tensor":
        return add(-self, other)

    def __neg__(self) -> "Tensor":
        return self * -1

    def __mul__(self, other: Number | "Tensor") -> "Tensor":
        return multiply(self, other)

    def __rmul__(self, other: Number | "Tensor") -> "Tensor":
        return multiply(self, other)

    def __truediv__(self, other: Number | "Tensor") -> "Tensor":
        if isinstance(other, (int, float)):
            if other == 0:
                raise ValueError("division by zero")
            return self * (1 / other)
        return self * tensor_reciprocal(other)

    def __rtruediv__(self, other: Number | "Tensor") -> "Tensor":
        if isinstance(other, (int, float)):
            return tensor_reciprocal(self) * other
        return other / self

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return matmul(self, other)

    def reshape(self, shape: tuple[int, ...]) -> "Tensor":
        return reshape(self, shape)

    def flatten(self, start_axis: int = 0) -> "Tensor":
        return flatten(self, start_axis=start_axis)

    def split(self, sizes: Sequence[int], axis: int = -1) -> list["Tensor"]:
        return split(self, sizes, axis=axis)

    def permute(self, axes: tuple[int, ...]) -> "Tensor":
        return permute(self, axes)

    def conv2d_valid(self, kernel: "Tensor", *, stride: int = 1) -> "Tensor":
        return conv2d_valid(self, kernel, stride=stride)

    def avg_pool2d(
        self,
        window: tuple[int, int],
        *,
        stride: tuple[int, int] | None = None,
    ) -> "Tensor":
        return avg_pool2d(self, window, stride=stride)

    def max_pool2d(
        self,
        window: tuple[int, int],
        *,
        stride: tuple[int, int] | None = None,
    ) -> "Tensor":
        return max_pool2d(self, window, stride=stride)

    @property
    def T(self) -> "Tensor":
        return transpose(self)

    def sum(self, axis: int | None = None) -> "Tensor":
        return tensor_sum(self, axis=axis)

    def mean(self, axis: int | None = None) -> "Tensor":
        return tensor_mean(self, axis=axis)

    def softmax(self, axis: int = -1) -> "Tensor":
        return tensor_softmax(self, axis=axis)

    def exp(self) -> "Tensor":
        return tensor_exp(self)

    def log(self) -> "Tensor":
        return tensor_log(self)

    def sqrt(self) -> "Tensor":
        return tensor_sqrt(self)

    def relu(self) -> "Tensor":
        return tensor_relu(self)

    def gelu(self) -> "Tensor":
        return tensor_gelu(self)

    def sigmoid(self) -> "Tensor":
        return tensor_sigmoid(self)

    def tanh(self) -> "Tensor":
        return tensor_tanh(self)

    def reciprocal(self) -> "Tensor":
        return tensor_reciprocal(self)

    def zero_grad(self) -> None:
        if self.requires_grad:
            # A sparse table's "zero" gradient is no rows at all.
            self.grad = None if self.sparse_grad else kernels.zeros(len(self.data))

    def backward(
        self,
        grad: Tensor | Sequence[Number] | None = None,
        *,
        retain_graph: bool = False,
        accumulate: bool = False,
    ) -> None:
        """Fill ``.grad`` on every tensor this one was computed from.

        Gradients are reset before the pass, so the leaves end up with this
        output's gradient rather than an accumulation over calls. With
        ``accumulate=True`` the leaves keep their gradients and this pass
        adds to them (as PyTorch always does): run backward on several
        micro-batches, then take one optimizer step.

        Afterwards the graph is freed (as in PyTorch): each intermediate
        result drops its backward function and its links to its inputs.
        Those links form reference cycles, so without this every step's
        graph would linger until Python's cycle collector ran. Pass
        ``retain_graph=True`` to call backward on the same graph again.
        """

        topo = _topological_order(self)

        # Start every gradient empty; the first _add_grad adopts its list.
        for node in topo:
            if node.requires_grad and (node._prev or not accumulate):
                node.grad = None

        _add_grad(self, _seed_grad(self, grad))

        for node in reversed(topo):
            if node.requires_grad:
                if _profiler is None:
                    node._backward()
                else:                                   # profiler.py is timing ops
                    start = perf_counter()
                    node._backward()
                    _profiler.backward(node._op, perf_counter() - start)

        # Leaves the output does not depend on get zeros, not None.
        for node in topo:
            if node.requires_grad and node.grad is None and not node._prev:
                node.zero_grad()

        if not retain_graph:
            for node in topo:
                if node._prev:
                    node._prev = set()
                    node._backward = _freed_backward


def _no_backward() -> None:
    return None


def _freed_backward() -> None:
    raise RuntimeError(
        "this graph was already freed by backward(); "
        "pass retain_graph=True to the first backward() to reuse it"
    )


def _topological_order(root: Tensor) -> list[Tensor]:
    """Children-before-parents order, without Python recursion limits."""

    order: list[Tensor] = []
    visited: set[Tensor] = set()
    pending: list[tuple[Tensor, bool]] = [(root, False)]

    while pending:
        node, children_done = pending.pop()
        if children_done:
            order.append(node)
            continue
        if node in visited:
            continue
        visited.add(node)
        pending.append((node, True))
        for child in node._prev:
            if child not in visited:
                pending.append((child, False))

    return order


# ---------------------------------------------------------------------------
# Recording ops
# ---------------------------------------------------------------------------


_grad_enabled = True
_profiler = None          # set by profiler.profile() while it measures ops


@contextmanager
def no_grad() -> Iterator[None]:
    """Run ops without recording the graph, e.g. for evaluation or sampling."""

    global _grad_enabled
    previous = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = previous


def is_grad_enabled() -> bool:
    return _grad_enabled


def _op_result(
    data: list[float],
    shape: tuple[int, ...],
    parents: tuple[Tensor, ...],
    op: str,
) -> Tensor:
    """Wrap the output of an op.

    This is the one place that decides whether an op joins the autograd
    graph: only when gradients are enabled and some parent needs them. Ops
    pass trusted floats, so no validation happens here. Assigning
    ``out.data`` stores them in the backend's form: a kernel's C array as
    is, a list built by Python code converted once.

    A custom op is three steps: compute ``data``, call ``_op_result``, then
    pass a function that reads ``out.grad`` and calls ``_add_grad`` on each
    parent that requires grad to ``_set_backward(out, function)``.
    """

    out = Tensor.__new__(Tensor)
    out.data = data
    out.shape = shape
    out.requires_grad = _grad_enabled and any(
        parent.requires_grad for parent in parents
    )
    out.grad = None
    out._prev = set(parents) if out.requires_grad else set()
    out._op = op
    out._backward = _no_backward
    if _profiler is not None:
        _profiler.forward(op)
    return out


def _set_backward(out: Tensor, backward: Callable[[], None]) -> None:
    """Record how to backpropagate through ``out``, if it needs gradients.

    The backward function is a closure that refers back to ``out``. That
    reference cycle is harmless during training (backward() breaks it), but
    under ``no_grad`` or for results nobody differentiates it would keep
    every activation alive until Python's cycle collector happened to run.
    That collector counts objects, not bytes, so a few big activation lists
    can pile up into gigabytes first.
    """

    if out.requires_grad:
        out._backward = backward


def _seed_grad(
    tensor: Tensor,
    grad: Tensor | Sequence[Number] | None,
) -> list[float]:
    if grad is None:
        if tensor.shape != (1,):
            raise ValueError("non-scalar tensor backward requires an explicit gradient")
        return [1.0]

    if isinstance(grad, Tensor):
        if grad.shape != tensor.shape:
            raise ValueError("backward gradient shape must match tensor shape")
        return grad.data[:]

    if len(grad) != len(tensor.data):
        raise ValueError("backward gradient length must match tensor data")
    return list(map(float, grad))


def _grad_data(tensor: Tensor) -> list[float]:
    if tensor.grad is None:
        return [0.0] * len(tensor.data)
    return tensor.grad


class RowGrad:
    """The gradient of a big table where only a few rows were used.

    An embedding table of 1,000,000 x 10 numbers, of which a batch reads
    3,000 rows, has a gradient that is zero everywhere else. Storing it
    dense means writing (and then applying) 10 million numbers per step;
    a RowGrad keeps just the row ids and their gradient rows, like
    PyTorch's sparse embedding gradients. The same id may appear more than
    once; its rows add up (``coalesced`` merges them).
    """

    def __init__(self, ids: Sequence[int], values: Sequence[float], width: int, rows: int) -> None:
        self.ids, self.values, self.width, self.rows = list(ids), kernels.storage(values), width, rows

    def __len__(self) -> int:                       # as long as the dense gradient it stands for
        return self.rows * self.width

    def plus(self, other: "RowGrad") -> "RowGrad":
        return RowGrad(self.ids + other.ids, list(self.values) + list(other.values), self.width, self.rows)

    def coalesced(self) -> "RowGrad":
        """Each id once, with its rows summed."""

        unique = list(dict.fromkeys(self.ids))
        position = {row: index for index, row in enumerate(unique)}
        summed = kernels.scatter_add_rows(self.values, kernels.index_table([position[i] for i in self.ids]),
                                          width=self.width, rows=len(unique))
        return RowGrad(unique, summed, self.width, self.rows)

    def to_dense(self) -> list[float]:
        return kernels.scatter_add_rows(self.values, kernels.index_table(self.ids), width=self.width, rows=self.rows)


def _add_grad(tensor: Tensor, grad: Sequence[Number]) -> None:
    """Accumulate ``grad`` into ``tensor.grad`` (gradients add up over uses)."""

    if len(grad) != len(tensor.data):
        raise ValueError("gradient length does not match tensor data")
    if isinstance(grad, RowGrad) or isinstance(tensor.grad, RowGrad):
        if tensor.grad is None:
            tensor.grad = grad
        elif isinstance(grad, RowGrad) and isinstance(tensor.grad, RowGrad):
            tensor.grad = tensor.grad.plus(grad)
        else:                                       # mixed with a dense gradient: go dense
            dense = [g.to_dense() if isinstance(g, RowGrad) else g for g in (tensor.grad, grad)]
            tensor.grad = kernels.add(*dense)
        return
    if tensor.grad is None:
        # Gradients are never modified in place (accumulating builds a new
        # sequence), so the first one can be adopted as-is, not copied.
        tensor.grad = grad
        return
    tensor.grad = kernels.add(tensor.grad, grad)


# ---------------------------------------------------------------------------
# Elementwise math with broadcasting
# ---------------------------------------------------------------------------


def add(left: Tensor, right: Number | Tensor) -> Tensor:
    """Elementwise ``left + right`` with numpy-style broadcasting.

    d(out)/d(left) = 1 and d(out)/d(right) = 1, so both sides receive the
    output gradient, summed back over any broadcast axes.
    """

    if isinstance(right, (int, float)):
        constant = kernels.storage([float(right)]) * len(left.data)    # repeated at C speed
        out = _op_result(kernels.add(left.data, constant), left.shape, (left,), "+")

        def _backward_scalar() -> None:
            _add_grad(left, _grad_data(out))

        _set_backward(out, _backward_scalar)
        return out

    out_shape = _broadcast_shape(left.shape, right.shape)
    left_values = _expand(left, out_shape)
    right_values = _expand(right, out_shape)
    out = _op_result(kernels.add(left_values, right_values), out_shape, (left, right), "+")

    def _backward() -> None:
        grad = _grad_data(out)
        if left.requires_grad:
            _add_grad(left, _reduce_to_shape(grad, out_shape, left.shape))
        if right.requires_grad:
            _add_grad(right, _reduce_to_shape(grad, out_shape, right.shape))

    _set_backward(out, _backward)
    return out


def subtract(left: Tensor, right: Tensor) -> Tensor:
    """Elementwise ``left - right`` with numpy-style broadcasting."""

    out_shape = _broadcast_shape(left.shape, right.shape)
    left_values = _expand(left, out_shape)
    right_values = _expand(right, out_shape)
    out = _op_result(kernels.add_scaled(left_values, right_values, -1.0), out_shape, (left, right), "-")

    def _backward() -> None:
        grad = _grad_data(out)
        if left.requires_grad:
            _add_grad(left, _reduce_to_shape(grad, out_shape, left.shape))
        if right.requires_grad:
            _add_grad(right, _reduce_to_shape(kernels.scale(grad, -1.0), out_shape, right.shape))

    _set_backward(out, _backward)
    return out


def multiply(left: Tensor, right: Number | Tensor) -> Tensor:
    """Elementwise ``left * right`` with numpy-style broadcasting.

    d(out)/d(left) = right and d(out)/d(right) = left.
    """

    if isinstance(right, (int, float)):
        out = _op_result(kernels.scale(left.data, right), left.shape, (left,), "*")

        def _backward_scalar() -> None:
            _add_grad(left, kernels.scale(_grad_data(out), right))

        _set_backward(out, _backward_scalar)
        return out

    out_shape = _broadcast_shape(left.shape, right.shape)
    left_values = _expand(left, out_shape)
    right_values = _expand(right, out_shape)
    out = _op_result(kernels.multiply(left_values, right_values), out_shape, (left, right), "*")

    def _backward() -> None:
        grad = _grad_data(out)
        if left.requires_grad:
            left_grad = kernels.multiply(grad, right_values)
            _add_grad(left, _reduce_to_shape(left_grad, out_shape, left.shape))
        if right.requires_grad:
            right_grad = kernels.multiply(grad, left_values)
            _add_grad(right, _reduce_to_shape(right_grad, out_shape, right.shape))

    _set_backward(out, _backward)
    return out


def _broadcast_shape(
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> tuple[int, ...]:
    shape = []
    max_ndim = max(len(left_shape), len(right_shape))
    padded_left = (1,) * (max_ndim - len(left_shape)) + left_shape
    padded_right = (1,) * (max_ndim - len(right_shape)) + right_shape

    for left_dim, right_dim in zip(padded_left, padded_right):
        if left_dim == right_dim:
            shape.append(left_dim)
        elif left_dim == 1:
            shape.append(right_dim)
        elif right_dim == 1:
            shape.append(left_dim)
        else:
            raise ValueError("tensor shapes cannot be broadcast")
    return tuple(shape)


@lru_cache(maxsize=256)
def _broadcast_index_map(
    out_shape: tuple[int, ...],
    source_shape: tuple[int, ...],
) -> tuple[int, ...]:
    """For each flat output position, the flat source position it reads.

    A broadcast axis is just an axis with stride 0: walking along it keeps
    reading the same source value.
    """

    padded_shape = (1,) * (len(out_shape) - len(source_shape)) + source_shape
    source_strides = _strides(padded_shape)
    strides = tuple(
        0 if source_dim == 1 else stride
        for source_dim, stride in zip(padded_shape, source_strides)
    )
    return _strided_indices(out_shape, strides)


@lru_cache(maxsize=256)
def _broadcast_index_table(out_shape: tuple[int, ...], source_shape: tuple[int, ...], backend: str):
    """``_broadcast_index_map`` as the active backend's index table, so a
    broadcast is one gather (and its gradient one scatter-add) kernel call."""

    return kernels.index_table(_broadcast_index_map(out_shape, source_shape))


def _repeated_block(
    out_shape: tuple[int, ...],
    source_shape: tuple[int, ...],
) -> tuple[int, int] | None:
    """``(outer, inner)`` if broadcasting is one block of the source repeated.

    The common broadcasts look like this: a bias (F,) over (N, F), or a conv
    bias (F, 1, 1) over (B, F, H, W). The source's non-1 axes are one
    contiguous run matching ``out_shape``, so the output is

        outer copies of [each source value repeated inner times]

    which list repetition builds at C speed. Other patterns return None.
    """

    padded = (1,) * (len(out_shape) - len(source_shape)) + source_shape
    kept = [axis for axis, dim in enumerate(padded) if dim != 1]
    if not kept or kept != list(range(kept[0], kept[-1] + 1)):
        return None
    if any(padded[axis] != out_shape[axis] for axis in kept):
        return None
    return _numel(out_shape[: kept[0]]), _numel(out_shape[kept[-1] + 1 :])


def _expand(tensor: Tensor, out_shape: tuple[int, ...]) -> list[float]:
    """Values of ``tensor`` laid out over ``out_shape`` (no copy if equal)."""

    if tensor.shape == out_shape:
        return tensor.data
    data = tensor.data
    block = _repeated_block(out_shape, tensor.shape)
    if block is not None:
        outer, inner = block
        if inner == 1:
            return data * outer
        # Repeat each value inner times by repeating length-1 slices, which
        # works at C speed for lists and arrays alike.
        repeated = data[:0]
        for index in range(len(data)):
            repeated += data[index : index + 1] * inner
        return repeated * outer
    table = _broadcast_index_table(out_shape, tensor.shape, kernels.BACKEND)
    return kernels.gather_indexed(data, table, batches=1, image_size=len(data))


def _reduce_to_shape(
    grad: list[float],
    out_shape: tuple[int, ...],
    source_shape: tuple[int, ...],
) -> list[float]:
    """Sum a broadcast gradient back down to the source tensor's shape."""

    if out_shape == source_shape:
        return grad
    block = _repeated_block(out_shape, source_shape)
    if block is not None:
        outer, inner = block
        return kernels.sum_repeated(grad, outer=outer, size=_numel(source_shape), inner=inner)
    table = _broadcast_index_table(out_shape, source_shape, kernels.BACKEND)
    return kernels.scatter_add_indexed(grad, table, batches=1, image_size=_numel(source_shape))


def _unary(
    tensor: Tensor,
    out_data: list[float],
    input_grad: Callable[[list[float]], list[float]],
    op: str,
) -> Tensor:
    """Wrap an elementwise op whose backward maps out.grad -> tensor grad."""

    out = _op_result(out_data, tensor.shape, (tensor,), op)

    def _backward() -> None:
        _add_grad(tensor, input_grad(_grad_data(out)))

    _set_backward(out, _backward)
    return out


def tensor_exp(tensor: Tensor) -> Tensor:
    """Elementwise exponential: y = e^x, dy/dx = y."""

    out_data = kernels.exp(tensor.data)
    return _unary(tensor, out_data, lambda grad: kernels.multiply(grad, out_data), "exp")


def tensor_sqrt(tensor: Tensor) -> Tensor:
    """Elementwise square root: y = sqrt(x), dy/dx = 1 / (2 y)."""

    if min(tensor.data) < 0:
        raise ValueError("sqrt is only defined for non-negative values")
    out_data = kernels.sqrt(tensor.data)
    return _unary(tensor, out_data, lambda grad: kernels.scale(kernels.multiply(grad, kernels.reciprocal(out_data)), 0.5),
                  "sqrt")


def tensor_log(tensor: Tensor) -> Tensor:
    """Elementwise natural logarithm: y = ln x, dy/dx = 1 / x."""

    if min(tensor.data) <= 0:
        raise ValueError("log is only defined for positive values")
    inputs = tensor.data
    return _unary(
        tensor,
        kernels.log(inputs),
        lambda grad: kernels.multiply(grad, kernels.reciprocal(inputs)),
        "log",
    )


def tensor_relu(tensor: Tensor) -> Tensor:
    """Elementwise rectified linear unit: y = max(0, x)."""

    inputs = tensor.data
    return _unary(
        tensor,
        kernels.relu(inputs),
        lambda grad: kernels.relu_backward(inputs, grad),
        "relu",
    )


def tensor_gelu(tensor: Tensor) -> Tensor:
    """Elementwise Gaussian error linear unit (exact erf form).

    y  = x * Phi(x)                 Phi = standard normal CDF
    dy = Phi(x) + x * phi(x)        phi = standard normal PDF
    """

    inputs = tensor.data
    out_data, cdf = kernels.gelu(inputs)   # cdf = Phi(x), kept for backward
    return _unary(
        tensor,
        out_data,
        lambda grad: kernels.gelu_backward(inputs, cdf, grad),
        "gelu",
    )


def tensor_sigmoid(tensor: Tensor) -> Tensor:
    """Elementwise sigmoid: y = 1 / (1 + e^-x), dy/dx = y (1 - y)."""

    out_data = kernels.sigmoid(tensor.data)
    return _unary(
        tensor,
        out_data,
        lambda grad: kernels.sigmoid_backward(out_data, grad),
        "sigmoid",
    )


def tensor_tanh(tensor: Tensor) -> Tensor:
    """Elementwise hyperbolic tangent: dy/dx = 1 - y^2."""

    out_data = kernels.tanh(tensor.data)
    return _unary(
        tensor,
        out_data,
        lambda grad: kernels.tanh_backward(out_data, grad),
        "tanh",
    )


def tensor_reciprocal(tensor: Tensor) -> Tensor:
    """Elementwise reciprocal: y = 1 / x, dy/dx = -y^2."""

    if 0.0 in tensor.data:
        raise ValueError("division by zero")
    out_data = kernels.reciprocal(tensor.data)
    squared = kernels.multiply(out_data, out_data)
    return _unary(tensor, out_data, lambda grad: kernels.scale(kernels.multiply(grad, squared), -1.0), "reciprocal")


# ---------------------------------------------------------------------------
# Matrix multiplication
# ---------------------------------------------------------------------------


# A C-speed dot product (math.sumprod on Python 3.12+); see kernels.py.
_dot = kernels.dot


def _rows(data: list[float], width: int) -> list[list[float]]:
    """Split a flat row-major matrix into its rows."""

    return [data[start : start + width] for start in range(0, len(data), width)]


def matmul(left: Tensor, right: Tensor) -> Tensor:
    """Matrix product following numpy's ``@`` shape rules.

    Shapes (``...`` is any number of leading batch axes):

    - (k,)        @ (k,)        -> (1,)        dot product
    - (n, k)      @ (k,)        -> (n,)        matrix-vector
    - (k,)        @ (k, m)      -> (m,)        vector-matrix
    - (..., n, k) @ (k, m)      -> (..., n, m) shared right matrix
    - (..., n, k) @ (..., k, m) -> (..., n, m) batched, same batch axes
    - (n, k)      @ (..., k, m) -> (..., n, m) shared left matrix

    For one batch entry, with C = A @ B:

        dA = dC @ B^T
        dB = A^T @ dC
    """

    plan = _matmul_plan(left.shape, right.shape)
    sizes = dict(batches=plan.batches, a_step=plan.left_step, b_step=plan.right_step)
    out_step = plan.rows * plan.cols

    data = kernels.gemm(
        left.data,
        right.data,
        m=plan.rows,
        k=plan.shared,
        n=plan.cols,
        c_step=out_step,
        **sizes,
    )
    out = _op_result(data, plan.out_shape, (left, right), "matmul")

    def _backward() -> None:
        grad = _grad_data(out)
        if left.requires_grad:
            # dA = dC @ B^T. A shared left matrix (step 0) sums over batches.
            _add_grad(left, kernels.gemm(
                grad,
                right.data,
                batches=plan.batches,
                m=plan.rows,
                k=plan.cols,
                n=plan.shared,
                a_step=out_step,
                b_step=plan.right_step,
                c_step=plan.left_step,
                trans_b=True,
            ))
        if right.requires_grad:
            # dB = A^T @ dC. A shared right matrix (step 0) sums over batches.
            _add_grad(right, kernels.gemm(
                left.data,
                grad,
                batches=plan.batches,
                m=plan.shared,
                k=plan.rows,
                n=plan.cols,
                a_step=plan.left_step,
                b_step=out_step,
                c_step=plan.right_step,
                trans_a=True,
            ))

    _set_backward(out, _backward)
    return out


class _MatmulPlan(NamedTuple):
    """How to walk a (possibly batched) product as ``batches`` 2D products."""

    batches: int
    rows: int
    shared: int
    cols: int
    left_step: int   # flat offset between batch entries; 0 = shared matrix
    right_step: int
    out_shape: tuple[int, ...]


def _matmul_plan(
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> _MatmulPlan:
    # Vectors become matrices: a left (k,) is a row (1, k), a right (k,) is a
    # column (k, 1). The extra axis is dropped again from the output shape.
    left_matrix = (1, *left_shape) if len(left_shape) == 1 else left_shape
    right_matrix = (*right_shape, 1) if len(right_shape) == 1 else right_shape
    rows, shared = left_matrix[-2:]
    right_shared, cols = right_matrix[-2:]
    if shared != right_shared:
        raise ValueError("matmul inner dimensions must match")

    left_batch = left_matrix[:-2]
    right_batch = right_matrix[:-2]
    if not right_batch:
        # A shared right matrix: fold every left batch axis into the rows.
        rows = _numel(left_matrix[:-1])
        batch_shape: tuple[int, ...] = ()
        out_shape = (*left_matrix[:-1], cols)
    elif not left_batch or left_batch == right_batch:
        batch_shape = right_batch
        out_shape = (*batch_shape, rows, cols)
    else:
        raise ValueError("matmul batch dimensions must match")

    if len(left_shape) == 1 and len(right_shape) == 1:
        out_shape = (1,)
    elif len(left_shape) == 1:
        out_shape = out_shape[:-2] + out_shape[-1:]
    elif len(right_shape) == 1:
        out_shape = out_shape[:-1]

    return _MatmulPlan(
        batches=_numel(batch_shape),
        rows=rows,
        shared=shared,
        cols=cols,
        left_step=rows * shared if left_batch else 0,
        right_step=shared * cols if right_batch else 0,
        out_shape=out_shape,
    )


def rotary_embedding(tensor: Tensor, *, start: int = 0, base: float = 10000.0, scale: float = 1.0) -> Tensor:
    """Rotary position embedding (RoPE, used by LLaMA) for (..., T, D) tensors.

    Instead of adding a learned position vector to each token, RoPE rotates
    every (even, odd) pair of a query or key by an angle that grows with the
    token's position p (starting at ``start``):

        angle = p * base^(-2i / D)          for pair i
        out[2i]   = x[2i] cos - x[2i+1] sin
        out[2i+1] = x[2i] sin + x[2i+1] cos

    A query-key dot product then depends only on how far apart the two
    tokens are. ``scale`` multiplies every position first: position
    interpolation (Chen et al., 2023) uses scale = trained context / new
    context, so a longer context reuses the angles the model was trained on
    instead of new, larger ones. Written with whole arrays: out = x * C + swap(x) * S, where
    swap exchanges the two values of each pair and S holds (-sin, +sin).
    A rotation's gradient is the opposite rotation:

        dx = g * C + swap(g * S)
    """

    length, width = tensor.shape[-2:]
    if width % 2:
        raise ValueError("rotary_embedding needs an even last axis")
    cos_table, sin_table = _rotary_tables(length, width, start, base, scale)
    repeats = tensor.numel // (length * width)
    cos_full = kernels.storage(cos_table * repeats)     # in the current precision
    sin_full = kernels.storage(sin_table * repeats)
    swap = _pair_swap_index(width)
    sizes = dict(batches=tensor.numel // width, image_size=width)

    swapped = kernels.gather_indexed(tensor.data, swap, **sizes)
    data = kernels.add(kernels.multiply(tensor.data, cos_full), kernels.multiply(swapped, sin_full))
    out = _op_result(data, tensor.shape, (tensor,), "rotary_embedding")

    def _backward() -> None:
        grad = _grad_data(out)
        rotated_back = kernels.gather_indexed(kernels.multiply(grad, sin_full), swap, **sizes)
        _add_grad(tensor, kernels.add(kernels.multiply(grad, cos_full), rotated_back))

    _set_backward(out, _backward)
    return out


@lru_cache(maxsize=64)
def _rotary_tables(length: int, width: int, start: int, base: float, scale: float = 1.0):
    """Flat (T x D) tables of cos and signed sin (-sin at even, +sin at odd)."""

    cos_values, sin_values = [], []
    for position in range(start, start + length):
        for pair in range(width // 2):
            angle = position * scale * base ** (-2 * pair / width)
            cos_values += [math.cos(angle), math.cos(angle)]
            sin_values += [-math.sin(angle), math.sin(angle)]
    return tuple(cos_values), tuple(sin_values)    # plain floats: valid in any precision


@lru_cache(maxsize=64)
def _pair_swap_index(width: int):
    """0, 1, 2, 3, ... -> 1, 0, 3, 2, ...: each value's partner in its pair."""

    return kernels.index_table(index ^ 1 for index in range(width))


# ---------------------------------------------------------------------------
# Shape ops
# ---------------------------------------------------------------------------


def reshape(tensor: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Return a tensor view-shaped copy with the same flat data order."""

    shape = tuple(shape)
    if _numel(shape) != len(tensor.data) or any(dim <= 0 for dim in shape):
        raise ValueError("reshape must preserve the number of elements")

    out = _op_result(tensor.data[:], shape, (tensor,), "reshape")

    def _backward() -> None:
        _add_grad(tensor, _grad_data(out))

    _set_backward(out, _backward)
    return out


def flatten(tensor: Tensor, start_axis: int = 0) -> Tensor:
    """Return tensor values flattened from start_axis onward."""

    start_axis = _normalize_axis(start_axis, tensor.ndim)
    leading_shape = tensor.shape[:start_axis]
    flattened = _numel(tensor.shape[start_axis:])
    return reshape(tensor, (*leading_shape, flattened))


def _integer_keys(key):
    """Turn integer-like keys (e.g. NumPy integers) into ints; reject bools."""

    def convert(item):
        if isinstance(item, bool):
            raise IndexError("tensor indices may not be booleans")
        if isinstance(item, (slice, type(Ellipsis))):
            return item
        try:
            return operator.index(item)
        except TypeError:
            raise IndexError("tensor indices must be integers, slices or ...") from None

    return tuple(convert(item) for item in key) if isinstance(key, tuple) else convert(key)


def take_slices(tensor: Tensor, key) -> Tensor:
    """NumPy-style basic indexing: integers, slices and one ``...``.

    An integer picks one position and drops that axis; a slice keeps the
    axis with the chosen positions. Which input values land in the output
    is a fixed list of flat offsets, so this is a gather through an index
    table, and the backward pass scatters the gradient back.
    """

    shape, chosen = _slice_plan(tensor.shape, key)
    index = _slice_index(tensor.shape, chosen)
    sizes = dict(batches=1, image_size=tensor.numel)
    out = _op_result(kernels.gather_indexed(tensor.data, index, **sizes), shape, (tensor,), "slice")

    def _backward() -> None:
        _add_grad(tensor, kernels.scatter_add_indexed(_grad_data(out), index, **sizes))

    _set_backward(out, _backward)
    return out


def _slice_plan(shape: tuple[int, ...], key) -> tuple[tuple[int, ...], tuple]:
    """(output shape, per axis: an int position or a tuple of positions)."""

    key = key if isinstance(key, tuple) else (key,)
    if key.count(Ellipsis) > 1:
        raise IndexError("an index can only have one ...")
    if Ellipsis in key:
        at = key.index(Ellipsis)
        fill = len(shape) - (len(key) - 1)
        key = key[:at] + (slice(None),) * fill + key[at + 1 :]
    if len(key) > len(shape):
        raise IndexError("too many indices for tensor")
    key = key + (slice(None),) * (len(shape) - len(key))

    out_shape, chosen = [], []
    for item, size in zip(key, shape):
        if isinstance(item, int):
            chosen.append(_normalize_index(item, size))
        elif isinstance(item, slice):
            positions = tuple(range(*item.indices(size)))
            if not positions:
                raise IndexError("a slice may not be empty")
            chosen.append(positions)
            out_shape.append(len(positions))
        else:
            raise IndexError("tensor indices must be integers, slices or ...")
    if not out_shape:
        out_shape = [1]                     # every axis picked by an integer
    return tuple(out_shape), tuple(chosen)


@lru_cache(maxsize=256)
def _slice_index(shape: tuple[int, ...], chosen: tuple):
    strides = _strides(shape)
    offsets = [0]
    for item, stride in zip(chosen, strides):
        steps = [item * stride] if isinstance(item, int) else [position * stride for position in item]
        offsets = [base + step for base in offsets for step in steps]
    return kernels.index_table(offsets)


def permute(tensor: Tensor, axes: tuple[int, ...]) -> Tensor:
    """Return a tensor with axes reordered: out axis i is input axis axes[i]."""

    axes = _normalize_axes(tuple(axes), tensor.ndim)
    out_shape = tuple(tensor.shape[axis] for axis in axes)
    source_indices = _permute_index_map(tensor.shape, axes)
    sizes = dict(batches=1, image_size=tensor.numel)
    out = _op_result(
        kernels.gather_indexed(tensor.data, source_indices, **sizes),
        out_shape,
        (tensor,),
        "permute",
    )

    def _backward() -> None:
        # A permutation moves every value exactly once, so just move back.
        _add_grad(tensor, kernels.scatter_add_indexed(_grad_data(out), source_indices, **sizes))

    _set_backward(out, _backward)
    return out


@lru_cache(maxsize=256)
def _permute_index_map(
    shape: tuple[int, ...],
    axes: tuple[int, ...],
) -> tuple[int, ...]:
    """For each flat output position, the flat input position it reads."""

    strides = _strides(shape)
    return kernels.index_table(_strided_indices(
        tuple(shape[axis] for axis in axes),
        tuple(strides[axis] for axis in axes),
    ))


def transpose(tensor: Tensor) -> Tensor:
    """Transpose a 2D tensor."""

    if len(tensor.shape) != 2:
        raise ValueError("transpose expects a 2D tensor")
    return permute(tensor, (1, 0))


def stack(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    """Join same-shaped tensors along a new axis."""

    if not tensors:
        raise ValueError("stack expects at least one tensor")

    base_shape = tensors[0].shape
    for tensor in tensors:
        if tensor.shape != base_shape:
            raise ValueError("stack expects all tensor shapes to match")

    axis = _normalize_insert_axis(axis, len(base_shape))
    out_shape = base_shape[:axis] + (len(tensors),) + base_shape[axis:]
    chunk = _numel(base_shape[axis:])
    return _join_chunks(tensors, out_shape, [chunk] * len(tensors), "stack")


def concat(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    """Join tensors along an existing axis; other dimensions must match."""

    if not tensors:
        raise ValueError("concat expects at least one tensor")

    base_shape = tensors[0].shape
    axis = _normalize_axis(axis, len(base_shape))
    for tensor in tensors:
        if (
            tensor.ndim != len(base_shape)
            or tensor.shape[:axis] != base_shape[:axis]
            or tensor.shape[axis + 1 :] != base_shape[axis + 1 :]
        ):
            raise ValueError("concat expects matching shapes off the concat axis")

    joined = sum(tensor.shape[axis] for tensor in tensors)
    out_shape = base_shape[:axis] + (joined,) + base_shape[axis + 1 :]
    chunks = [_numel(tensor.shape[axis:]) for tensor in tensors]
    return _join_chunks(tensors, out_shape, chunks, "concat")


def split(tensor: Tensor, sizes: Sequence[int], axis: int = -1) -> list[Tensor]:
    """Cut one axis into consecutive pieces of the given sizes (undoes concat).

    For every outer index (the axes before ``axis``), piece p is one
    contiguous chunk of the input row, so each piece is a gather through an
    index table of its chunk, and its gradient scatters back into place.

    All pieces hang off one shared node whose gradient buffer they add into
    in place; that node passes the finished buffer on once. (Scattering each
    piece into its own full-size buffer would cost a full copy per piece,
    which for an LSTM's time steps grows with the square of the length.)
    """

    axis = _normalize_axis(axis, tensor.ndim)
    if sum(sizes) != tensor.shape[axis] or any(size <= 0 for size in sizes):
        raise ValueError("split sizes must be positive and add up to the axis length")
    inner = _numel(tensor.shape[axis + 1 :])
    full = tensor.shape[axis] * inner                       # one outer index's values
    outer = tensor.numel // full

    whole = _op_result(tensor.data, tensor.shape, (tensor,), "split")

    def _backward_whole() -> None:
        _add_grad(tensor, _grad_data(whole))

    _set_backward(whole, _backward_whole)

    pieces = []
    offset = 0
    for size in sizes:
        index = _range_index(offset * inner, size * inner)
        out_shape = tensor.shape[:axis] + (size,) + tensor.shape[axis + 1 :]
        piece = _op_result(
            kernels.gather_indexed(tensor.data, index, batches=outer, image_size=full),
            out_shape, (whole,), "split_piece",
        )
        _set_backward(piece, _split_piece_backward(whole, piece, index, outer, full))
        pieces.append(piece)
        offset += size
    return pieces


def _split_piece_backward(whole: Tensor, piece: Tensor, index, outer: int, full: int) -> Callable[[], None]:
    def _backward() -> None:
        if whole.grad is None:
            whole.grad = kernels.zeros(whole.numel)   # private to this split
        kernels.scatter_add_indexed(_grad_data(piece), index, batches=outer, image_size=full, into=whole.grad)

    return _backward


@lru_cache(maxsize=2048)          # one entry per piece offset: long sequences need many
def _range_index(start: int, count: int):
    return kernels.index_table(range(start, start + count))


def _join_chunks(
    tensors: Sequence[Tensor],
    out_shape: tuple[int, ...],
    chunks: list[int],
    op: str,
) -> Tensor:
    """Interleave equal-count chunks: for every outer index, one chunk each.

    Stacking or concatenating along ``axis`` means: walk the axes before
    ``axis`` (the "outer" index), and for each one copy a contiguous chunk
    from every input in turn.
    """

    # Each input's chunk sits at a fixed offset inside every output row of
    # sum(chunks) values: copying it there is a scatter through an index
    # table of that range, and its gradient is the matching gather.
    outer = len(tensors[0].data) // chunks[0]
    full = sum(chunks)
    indices, offset = [], 0
    for chunk in chunks:
        indices.append(_range_index(offset, chunk))
        offset += chunk
    data = kernels.zeros(outer * full)
    for tensor, index in zip(tensors, indices):
        kernels.scatter_add_indexed(tensor.data, index, batches=outer, image_size=full, into=data)

    out = _op_result(data, out_shape, tuple(tensors), op)

    def _backward() -> None:
        grad = _grad_data(out)
        for tensor, index in zip(tensors, indices):
            if tensor.requires_grad:
                _add_grad(tensor, kernels.gather_indexed(grad, index, batches=outer, image_size=full))

    _set_backward(out, _backward)
    return out



def dropout(tensor: Tensor, p: float, rng: random.Random) -> Tensor:
    """Zero each value with probability ``p``; scale survivors by 1 / (1 - p).

    The scaling keeps every value's expected output equal to its input, so
    at evaluation time dropout is simply switched off (no rescaling). The
    backward pass sends gradient only through the values that survived,
    with the same 1 / (1 - p) factor.
    """

    if not 0.0 <= p < 1.0:
        raise ValueError("dropout probability must be in [0, 1)")
    if p == 0.0:
        return tensor

    mask = kernels.dropout_mask(tensor.numel, p, rng.getrandbits(64))
    out = _op_result(kernels.multiply(tensor.data, mask), tensor.shape, (tensor,), "dropout")

    def _backward() -> None:
        _add_grad(tensor, kernels.multiply(_grad_data(out), mask))

    _set_backward(out, _backward)
    return out


def gather_rows(table: Tensor, indices: Sequence[int]) -> Tensor:
    """Pick rows of a 2D table: out[i] = table[indices[i]].

    This is an embedding lookup. The backward pass adds each output-row
    gradient into the table row it came from (repeated ids accumulate).
    """

    if table.ndim != 2:
        raise ValueError("gather_rows expects a 2D table")
    if not indices:
        raise ValueError("gather_rows needs at least one index")

    rows, width = table.shape
    if min(indices) < 0 or max(indices) >= rows:
        raise ValueError("embedding index out of range")

    ids = kernels.index_table(indices)
    data = kernels.gather_rows(table.data, ids, width=width)
    out = _op_result(data, (len(indices), width), (table,), "gather_rows")

    def _backward() -> None:
        if table.sparse_grad:                       # only the rows that were read
            _add_grad(table, RowGrad(indices, _grad_data(out), width, rows))
        else:
            _add_grad(table, kernels.scatter_add_rows(_grad_data(out), ids, width=width, rows=rows))

    _set_backward(out, _backward)
    return out


class SparseMatrix:
    """A fixed matrix stored by its nonzero entries, row by row (CSR).

    For a graph's adjacency matrix almost every entry is zero: Cora has
    2,708 nodes but only 13,264 nonzeros of 7.3 million. Keeping only
    (row, column, value) triples makes a product with it cost one row
    operation per nonzero. The transpose is stored too, for the backward
    pass. The matrix is a constant (no gradient for its values).
    """

    def __init__(self, shape: tuple[int, int], entries: Sequence[tuple[int, int, float]]) -> None:
        self.shape = shape
        self.nnz = len(entries)
        self.row_start, self.cols, self.values = _csr(shape[0], entries)
        self.t_row_start, self.t_cols, self.t_values = _csr(shape[1], [(c, r, v) for r, c, v in entries])

    def to_dense(self) -> list[float]:
        rows, cols = self.shape
        out = [0.0] * (rows * cols)
        for r in range(rows):
            for e in range(self.row_start[r], self.row_start[r + 1]):
                out[r * cols + self.cols[e]] += self.values[e]
        return out


def _csr(rows: int, entries) -> tuple:
    ordered = sorted(entries)
    row_start = [0] * (rows + 1)
    for r, _, _ in ordered:
        row_start[r + 1] += 1
    for r in range(rows):
        row_start[r + 1] += row_start[r]
    return (kernels.index_table(row_start), kernels.index_table([c for _, c, _ in ordered]),
            kernels.storage([v for _, _, v in ordered]))


def sparse_matmul(matrix: SparseMatrix, dense: Tensor) -> Tensor:
    """(rows x inner) sparse @ (inner x n) dense -> (rows x n).

        out = A @ X        dX = A^T @ dout
    """

    rows, inner = matrix.shape
    if dense.ndim != 2 or dense.shape[0] != inner:
        raise ValueError("sparse_matmul needs a dense (inner, n) matrix")
    n = dense.shape[1]
    data = kernels.csr_matmul(matrix.row_start, matrix.cols, matrix.values, dense.data, rows=rows, n=n)
    out = _op_result(data, (rows, n), (dense,), "sparse_matmul")

    def _backward() -> None:
        _add_grad(dense, kernels.csr_matmul(matrix.t_row_start, matrix.t_cols, matrix.t_values,
                                            _grad_data(out), rows=inner, n=n))

    _set_backward(out, _backward)
    return out


def scatter_add_rows(values: Tensor, indices: Sequence[int], rows: int) -> Tensor:
    """The reverse of ``gather_rows``: a (rows, width) tensor of zeros with
    row i of ``values`` added into row ``indices[i]`` (repeats add up).

    Mixture-of-experts layers use it to put each expert's outputs back at
    the tokens they came from. The backward pass is a gather.
    """

    if values.ndim != 2 or len(indices) != values.shape[0]:
        raise ValueError("scatter_add_rows expects (len(indices), width) values")
    if indices and (min(indices) < 0 or max(indices) >= rows):
        raise ValueError("scatter index out of range")

    width = values.shape[1]
    ids = kernels.index_table(indices)
    data = kernels.scatter_add_rows(values.data, ids, width=width, rows=rows)
    out = _op_result(data, (rows, width), (values,), "scatter_add_rows")

    def _backward() -> None:
        if indices:
            _add_grad(values, kernels.gather_rows(_grad_data(out), ids, width=width))

    _set_backward(out, _backward)
    return out


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


def tensor_sum(tensor: Tensor, axis: int | None = None) -> Tensor:
    """Sum all values or reduce a tensor along one axis."""

    if axis is None:
        out = _op_result([sum(tensor.data)], (1,), (tensor,), "sum")

        def _backward_all() -> None:
            _add_grad(tensor, [_grad_data(out)[0]] * len(tensor.data))

        _set_backward(out, _backward_all)
        return out

    # Summing over an axis undoes broadcasting along it: the same reduction
    # as a broadcast gradient, and its backward is that broadcast.
    axis = _normalize_axis(axis, len(tensor.shape))
    kept = tensor.shape[:axis] + (1,) + tensor.shape[axis + 1 :]
    out = _op_result(
        _reduce_to_shape(tensor.data, tensor.shape, kept),
        _reduced_shape(tensor.shape, axis),
        (tensor,),
        "sum",
    )

    def _backward() -> None:
        _add_grad(tensor, _expand(Tensor(_grad_data(out), kept), tensor.shape))

    _set_backward(out, _backward)
    return out


def tensor_mean(tensor: Tensor, axis: int | None = None) -> Tensor:
    """Mean of all values or a tensor along one axis."""

    if axis is None:
        return tensor_sum(tensor) * (1 / len(tensor.data))

    axis = _normalize_axis(axis, len(tensor.shape))
    return tensor_sum(tensor, axis=axis) * (1 / tensor.shape[axis])


def tensor_softmax(tensor: Tensor, axis: int = -1) -> Tensor:
    """Softmax over one tensor axis.

    For one lane:  y_i = e^(x_i - max) / sum_j e^(x_j - max)
    Backward:      dx_i = y_i * (dy_i - sum_j dy_j * y_j)
    """

    axis = _normalize_axis(axis, len(tensor.shape))
    if axis == tensor.ndim - 1:
        # Last axis: every lane is a contiguous row (the attention case).
        return _softmax_last_axis(tensor)

    lanes = _axis_lanes(tensor.shape, axis)
    source = tensor.data
    data = [0.0] * tensor.numel
    for lane in lanes:
        values = source[lane]
        row_max = max(values)
        exp_values = [math.exp(value - row_max) for value in values]
        exp_sum = sum(exp_values)
        data[lane] = [value / exp_sum for value in exp_values]

    out = _op_result(data, tensor.shape, (tensor,), "softmax")

    def _backward() -> None:
        grad = _grad_data(out)
        tensor_grad = [0.0] * tensor.numel
        for lane in lanes:
            probabilities = data[lane]
            lane_grad = grad[lane]
            weighted_grad = _dot(lane_grad, probabilities)
            tensor_grad[lane] = [
                y * (g - weighted_grad)
                for y, g in zip(probabilities, lane_grad)
            ]
        _add_grad(tensor, tensor_grad)

    _set_backward(out, _backward)
    return out


def _softmax_last_axis(tensor: Tensor) -> Tensor:
    width = tensor.shape[-1]
    data = kernels.softmax_rows(tensor.data, width)
    out = _op_result(data, tensor.shape, (tensor,), "softmax")

    def _backward() -> None:
        _add_grad(tensor, kernels.softmax_rows_backward(data, _grad_data(out), width))

    _set_backward(out, _backward)
    return out


@lru_cache(maxsize=256)
def _axis_lanes(shape: tuple[int, ...], axis: int) -> tuple[slice, ...]:
    """One slice per "lane" along ``axis``.

    A lane is every value that differs only in its ``axis`` coordinate, e.g.
    one row when ``axis`` is the last axis. Lanes come out in row-major order
    of the remaining axes, which is exactly the reduced output order.
    """

    size = shape[axis]
    inner = _numel(shape[axis + 1 :])
    outer = _numel(shape[:axis])
    block = size * inner
    return tuple(
        slice(outer_index * block + inner_index, (outer_index + 1) * block, inner)
        for outer_index in range(outer)
        for inner_index in range(inner)
    )


# ---------------------------------------------------------------------------
# Convolution and pooling
# ---------------------------------------------------------------------------


def conv2d(image: Tensor, kernel: Tensor, *, stride: int = 1, padding: int | tuple[int, int] = 0) -> Tensor:
    """Convolution with zero ``padding`` around the image and a ``stride``.

    Same shapes and convention as ``conv2d_valid``; padding just adds a
    border of zeros first (``pad2d``). With kernel size k, padding p and
    stride s, an H-pixel side becomes (H + 2p - k) // s + 1 pixels: a 3x3
    kernel with padding 1 keeps the size, and stride 2 halves it.
    """

    padded = pad2d(image, padding) if padding else image
    return conv2d_valid(padded, kernel, stride=stride)


def pad2d(image: Tensor, padding: int | tuple[int, int]) -> Tensor:
    """Add ``padding`` zeros on every side of the last two axes (or
    ``(rows, cols)``: that many above and below, and left and right).

    Every input pixel moves to a fixed place in the larger zero image, listed
    in an index table, so padding is a scatter into zeros and its backward
    pass is the matching gather (the same kernels im2col uses).
    """

    pad_rows, pad_cols = (padding, padding) if isinstance(padding, int) else padding
    if pad_rows < 0 or pad_cols < 0:
        raise ValueError("padding must be non-negative")
    if image.ndim < 2:
        raise ValueError("pad2d expects at least a 2D tensor")
    rows, cols = image.shape[-2:]
    out_shape = (*image.shape[:-2], rows + 2 * pad_rows, cols + 2 * pad_cols)
    index = _pad2d_index(rows, cols, pad_rows, pad_cols)
    sizes = dict(batches=image.numel // (rows * cols), image_size=out_shape[-2] * out_shape[-1])
    out = _op_result(kernels.scatter_add_indexed(image.data, index, **sizes), out_shape, (image,), "pad2d")

    def _backward() -> None:
        _add_grad(image, kernels.gather_indexed(_grad_data(out), index, **sizes))

    _set_backward(out, _backward)
    return out


@lru_cache(maxsize=64)
def _pad2d_index(rows: int, cols: int, pad_rows: int, pad_cols: int) -> tuple[int, ...]:
    """Where each pixel of a (rows, cols) plane lands in the padded plane."""

    padded_cols = cols + 2 * pad_cols
    return kernels.index_table(
        (row + pad_rows) * padded_cols + col + pad_cols
        for row in range(rows)
        for col in range(cols)
    )


def upsample_nearest2d(image: Tensor, factor: int = 2) -> Tensor:
    """Make the last two axes ``factor`` times larger by repeating pixels.

    out[r, c] = image[r // factor, c // factor]: a gather through an index
    table, so the backward pass scatters each output gradient back onto the
    pixel it copied (every input pixel collects factor * factor of them).
    """

    if factor < 1:
        raise ValueError("upsample factor must be at least 1")
    rows, cols = image.shape[-2:]
    out_shape = (*image.shape[:-2], rows * factor, cols * factor)
    index = _upsample_index(rows, cols, factor)
    sizes = dict(batches=image.numel // (rows * cols), image_size=rows * cols)
    out = _op_result(kernels.gather_indexed(image.data, index, **sizes), out_shape, (image,), "upsample")

    def _backward() -> None:
        _add_grad(image, kernels.scatter_add_indexed(_grad_data(out), index, **sizes))

    _set_backward(out, _backward)
    return out


@lru_cache(maxsize=64)
def _upsample_index(rows: int, cols: int, factor: int):
    return kernels.index_table(
        (row // factor) * cols + col // factor
        for row in range(rows * factor)
        for col in range(cols * factor)
    )


def conv_transpose2d(image: Tensor, kernel: Tensor, *, stride: int = 1, padding: int = 0) -> Tensor:
    """Transposed convolution: image (B, Cin, H, W), kernel (Cin, Cout, kh, kw)
    -> (B, Cout, (H - 1) * stride - 2 * padding + kh, ...), as PyTorch's
    ``conv_transpose2d``. It grows an image: generators and decoders use it.

    It is exactly the backward pass of an ordinary convolution with the same
    kernel (Cin filters over Cout channels): each input pixel scatters
    kernel-weighted copies of itself onto a (kh x kw) patch of the output,
    and overlapping patches add up. So its three products are the three
    convolution kernels with their roles swapped:

        forward              = convolution's input gradient (col2im)
        gradient for image   = an ordinary convolution of the output gradient
        gradient for kernel  = convolution's kernel gradient, image and grad swapped
    """

    if image.ndim != 4 or kernel.ndim != 4 or kernel.shape[0] != image.shape[1]:
        raise ValueError("conv_transpose2d expects image (B, Cin, H, W) and kernel (Cin, Cout, kh, kw)")
    batches, channels_in, rows, cols = image.shape
    _, channels_out, kernel_rows, kernel_cols = kernel.shape
    full_rows, full_cols = (rows - 1) * stride + kernel_rows, (cols - 1) * stride + kernel_cols
    index = _conv2d_patch_index(channels_out, full_rows, full_cols, kernel_rows, kernel_cols, stride)
    sizes = dict(batches=batches, image_size=channels_out * full_rows * full_cols, filters=channels_in,
                 patch_size=channels_out * kernel_rows * kernel_cols, positions=rows * cols)
    data = kernels.conv_input_grad(image.data, kernel.data, index, **sizes)
    full = _op_result(data, (batches, channels_out, full_rows, full_cols), (image, kernel), "conv_transpose2d")

    def _backward() -> None:
        grad = _grad_data(full)
        if image.requires_grad:
            _add_grad(image, kernels.conv_forward(grad, index, kernel.data, **sizes))
        if kernel.requires_grad:
            _add_grad(kernel, kernels.conv_kernel_grad(grad, index, image.data, **sizes))

    _set_backward(full, _backward)
    if padding == 0:
        return full
    return full[:, :, padding : full_rows - padding, padding : full_cols - padding]


def depthwise_conv2d(image: Tensor, kernel: Tensor, *, stride: int = 1, padding: int = 0) -> Tensor:
    """Depthwise convolution: image (B, C, H, W), kernel (C, 1, kh, kw).

    Channel c of the output is channel c of the image convolved with
    kernel c alone (PyTorch's ``conv2d`` with ``groups=C``). It costs C
    times less than a full convolution with C filters; MobileNet pairs it
    with a 1x1 convolution that mixes the channels.
    """

    if image.ndim != 4 or kernel.ndim != 4 or kernel.shape[1] != 1 or kernel.shape[0] != image.shape[1]:
        raise ValueError("depthwise_conv2d expects image (B, C, H, W) and kernel (C, 1, kh, kw)")
    if padding:
        image = pad2d(image, padding)
    batches, channels, rows, cols = image.shape
    kernel_rows, kernel_cols = kernel.shape[2:]
    index = _conv2d_patch_index(1, rows, cols, kernel_rows, kernel_cols, stride)   # one channel's patches
    q_size = kernel_rows * kernel_cols
    out_rows, out_cols = (rows - kernel_rows) // stride + 1, (cols - kernel_cols) // stride + 1
    sizes = dict(batches=batches, channels=channels, channel_size=rows * cols, q_size=q_size)
    data = kernels.depthwise_forward(image.data, index, kernel.data, **sizes)
    out = _op_result(data, (batches, channels, out_rows, out_cols), (image, kernel), "depthwise_conv2d")

    def _backward() -> None:
        dkernel, dimage = kernels.depthwise_backward(image.data, index, kernel.data, _grad_data(out), **sizes)
        if kernel.requires_grad:
            _add_grad(kernel, dkernel)
        if image.requires_grad:
            _add_grad(image, dimage)

    _set_backward(out, _backward)
    return out


def conv2d_valid(image: Tensor, kernel: Tensor, *, stride: int = 1) -> Tensor:
    """Apply a valid (no padding) convolution-style filter.

    This uses the cross-correlation convention common in neural networks:
    the kernel is slid over the image without flipping it.

    Accepted shapes, with B batch, C input channels, O output filters:

    - image (H, W)       kernel (kh, kw)       -> (oh, ow)
    - image (H, W)       kernel (O, kh, kw)    -> (O, oh, ow)
    - image (C, H, W)    kernel (C, kh, kw)    -> (oh, ow)
    - image (C, H, W)    kernel (O, C, kh, kw) -> (O, oh, ow)

    Unbatched outputs with a single filter (O = 1) drop the filter axis.
    - image (B, C, H, W) kernel (O, C, kh, kw) -> (B, O, oh, ow)

    The kernel moves ``stride`` pixels at a time, so oh = (H - kh) // stride + 1.

    Internally everything is treated as the batched 4D case (reshaping a flat
    row-major list is free), using im2col:

        patches[p] = the C*kh*kw image values under the kernel at position p
        out[b, o, p] = kernel_o . patches[p]     i.e.  out_b = K @ patches_b^T
    """

    batches, channels, image_rows, image_cols, filters, out_shape = (
        _conv2d_shapes(image.shape, kernel.shape, stride)
    )
    kernel_rows, kernel_cols = kernel.shape[-2:]
    patch_index = _conv2d_patch_index(
        channels,
        image_rows,
        image_cols,
        kernel_rows,
        kernel_cols,
        stride,
    )
    patch_size = channels * kernel_rows * kernel_cols      # K
    positions = len(patch_index) // patch_size             # P
    image_size = channels * image_rows * image_cols

    # out_b = kernels (O x K) @ patches_b^T  ->  (O x P) per image. The C
    # kernel reads every patch straight from the image through the index
    # table (no patch matrix is built for the forward pass).
    data = kernels.conv_forward(
        image.data, patch_index, kernel.data, batches=batches, image_size=image_size,
        filters=filters, patch_size=patch_size, positions=positions,
    )
    out = _op_result(data, out_shape, (image, kernel), "conv2d_valid")

    def _backward() -> None:
        grad = _grad_data(out)
        if kernel.requires_grad:
            # dK = sum over images of dOut_b (O x P) @ patches_b (P x K),
            # patches again read through the index table.
            _add_grad(kernel, kernels.conv_kernel_grad(
                image.data, patch_index, grad, batches=batches, image_size=image_size,
                filters=filters, patch_size=patch_size, positions=positions,
            ))
        # Large stride-1 images: the gradient as a full convolution (faster there; on
        # small images the padding and flipping cost more than the scatter they save).
        if image.requires_grad and stride == 1 and kernel.ndim == 4 and image_rows * image_cols >= 4096:
            _add_grad(image, _conv2d_input_grad_full(grad, kernel, batches, image_rows, image_cols, out_shape))
        elif image.requires_grad:
            # dPatches_b = dOut_b^T (P x O) @ K (O x K), then col2im: add each
            # patch gradient back onto the pixels it was read from.
            _add_grad(image, kernels.conv_input_grad(
                grad, kernel.data, patch_index, batches=batches, image_size=image_size,
                filters=filters, patch_size=patch_size, positions=positions,
            ))

    _set_backward(out, _backward)
    return out


def _conv2d_input_grad_full(grad, kernel: Tensor, batches: int, image_rows: int, image_cols: int, out_shape):
    """The image gradient of a stride-1 convolution, as another convolution.

    Pixel (y, x) fed output (y - i, x - j) through kernel weight (i, j), so

        dImage[c, y, x] = sum over f, i, j of dOut[f, y - i, x - j] * K[f, c, i, j]

    which is a "full" convolution: pad dOut with kh - 1 / kw - 1 zeros on each
    side and slide the kernel flipped in both directions, with its filter and
    channel axes swapped. It runs on the forward kernel (patch size F*kh*kw)
    instead of a product over just F followed by a scatter (col2im).
    """

    filters, channels, kernel_rows, kernel_cols = kernel.shape
    out_rows, out_cols = out_shape[-2:]
    pad_rows, pad_cols = kernel_rows - 1, kernel_cols - 1
    padded_rows, padded_cols = out_rows + 2 * pad_rows, out_cols + 2 * pad_cols
    padded = kernels.scatter_add_indexed(
        grad, _pad2d_index(out_rows, out_cols, pad_rows, pad_cols),
        batches=batches * filters, image_size=padded_rows * padded_cols,
    )
    k = kernel.data
    flipped = kernels.storage([                                  # (C, F, kh, kw), rotated 180 degrees
        k[((f * channels + c) * kernel_rows + kernel_rows - 1 - i) * kernel_cols + kernel_cols - 1 - j]
        for c in range(channels) for f in range(filters) for i in range(kernel_rows) for j in range(kernel_cols)
    ])
    index = _conv2d_patch_index(filters, padded_rows, padded_cols, kernel_rows, kernel_cols, 1)
    return kernels.conv_forward(
        padded, index, flipped, batches=batches, image_size=filters * padded_rows * padded_cols,
        filters=channels, patch_size=filters * kernel_rows * kernel_cols, positions=image_rows * image_cols,
    )


def _conv2d_shapes(
    image_shape: tuple[int, ...],
    kernel_shape: tuple[int, ...],
    stride: int = 1,
) -> tuple[int, int, int, int, int, tuple[int, ...]]:
    """Validate shapes and return (B, C, H, W, O, output shape)."""

    mismatch = "conv2d_valid kernel shape does not match image shape"
    if len(image_shape) == 2:
        batches, channels = 1, 1
        image_rows, image_cols = image_shape
        if len(kernel_shape) == 2:
            filters = 1
        elif len(kernel_shape) == 3:
            filters = kernel_shape[0]
        else:
            raise ValueError(mismatch)
        kernel_channels = 1
    elif len(image_shape) == 3:
        batches = 1
        channels, image_rows, image_cols = image_shape
        if len(kernel_shape) == 3:
            filters = 1
            kernel_channels = kernel_shape[0]
        elif len(kernel_shape) == 4:
            filters, kernel_channels = kernel_shape[:2]
        else:
            raise ValueError(mismatch)
    elif len(image_shape) == 4:
        batches, channels, image_rows, image_cols = image_shape
        if len(kernel_shape) != 4:
            raise ValueError(mismatch)
        filters, kernel_channels = kernel_shape[:2]
    else:
        raise ValueError("conv2d_valid expects a 2D, 3D, or 4D image tensor")

    if channels != kernel_channels:
        raise ValueError("conv2d_valid image and kernel channels must match")

    kernel_rows, kernel_cols = kernel_shape[-2:]
    if kernel_rows > image_rows or kernel_cols > image_cols:
        raise ValueError("conv2d_valid kernel must fit inside the image")

    if stride <= 0:
        raise ValueError("conv2d stride must be positive")
    out_rows = (image_rows - kernel_rows) // stride + 1
    out_cols = (image_cols - kernel_cols) // stride + 1
    if len(image_shape) == 4:
        out_shape = (batches, filters, out_rows, out_cols)
    elif filters == 1:
        # Unbatched single-filter output drops the filter axis.
        out_shape = (out_rows, out_cols)
    else:
        out_shape = (filters, out_rows, out_cols)
    return batches, channels, image_rows, image_cols, filters, out_shape


def _size_bounded_cache(max_numbers: int):
    """Like ``lru_cache``, but bounded by the total length of the cached
    tables: a convolution's patch table holds positions x patch size numbers,
    hundreds of MB for a large image, and every new image size adds one."""

    def decorate(function):
        cache: OrderedDict = OrderedDict()
        total = 0

        @wraps(function)
        def cached(*args):
            nonlocal total
            if args in cache:
                cache.move_to_end(args)
                return cache[args]
            value = function(*args)
            cache[args] = value
            total += len(value)
            while total > max_numbers and len(cache) > 1:          # drop the least recently used
                _, old = cache.popitem(last=False)
                total -= len(old)
            return value

        cached.cache_clear = lambda: (cache.clear(), None)[1]
        return cached

    return decorate


@_size_bounded_cache(64_000_000)                   # 64M indices: 512 MB
def _conv2d_patch_index(
    channels: int,
    image_rows: int,
    image_cols: int,
    kernel_rows: int,
    kernel_cols: int,
    stride: int = 1,
) -> tuple[int, ...]:
    """Flat image indices of every patch, patch after patch (P x K, row-major).

    Index order inside a patch matches a flattened (C, kh, kw) kernel.
    Patches start every ``stride`` pixels.
    """

    channel_size = image_rows * image_cols
    patch_offsets = [
        channel * channel_size + row * image_cols + col
        for channel in range(channels)
        for row in range(kernel_rows)
        for col in range(kernel_cols)
    ]
    return kernels.index_table(
        out_row * stride * image_cols + out_col * stride + offset
        for out_row in range((image_rows - kernel_rows) // stride + 1)
        for out_col in range((image_cols - kernel_cols) // stride + 1)
        for offset in patch_offsets
    )


def avg_pool2d(
    image: Tensor,
    window: tuple[int, int],
    *,
    stride: tuple[int, int] | None = None,
) -> Tensor:
    """Average-pool a 2D tensor or each channel in a 3D/4D tensor.

    Like convolution, pooling reads the image through an index table that
    lists every window's pixels (see ``_pool2d_window_index``):

        out[plane, w] = (1 / n) * sum of the n pixels in window w

    and the backward pass hands each pixel 1/n of its window's gradient.
    """

    planes, out_shape, stride = _pool2d_setup("avg_pool2d", image, window, stride)
    window = tuple(window)   # a hashable key for the cached index table
    sizes = dict(
        planes=planes,
        plane_size=image.shape[-2] * image.shape[-1],
        window_size=window[0] * window[1],
    )
    index = _pool2d_window_index(image.shape[-2:], window, stride)
    out = _op_result(kernels.window_mean(image.data, index, **sizes), out_shape, (image,), "avg_pool2d")

    def _backward() -> None:
        _add_grad(image, kernels.window_mean_backward(_grad_data(out), index, **sizes))

    _set_backward(out, _backward)
    return out


def max_pool2d(
    image: Tensor,
    window: tuple[int, int],
    *,
    stride: tuple[int, int] | None = None,
) -> Tensor:
    """Max-pool a 2D tensor or each channel in a 3D/4D tensor.

    If values tie inside a pooling window, the gradient is routed to the first
    maximum in row-major order.
    """

    planes, out_shape, stride = _pool2d_setup("max_pool2d", image, window, stride)
    window = tuple(window)   # a hashable key for the cached index table
    index = _pool2d_window_index(image.shape[-2:], window, stride)
    data, max_indices = kernels.window_max(
        image.data, index,
        planes=planes,
        plane_size=image.shape[-2] * image.shape[-1],
        window_size=window[0] * window[1],
    )
    out = _op_result(data, out_shape, (image,), "max_pool2d")

    def _backward() -> None:
        # Only the winning pixel of each window affected the output: send
        # each window's gradient back to the pixel at its argmax.
        _add_grad(image, kernels.scatter_add_indexed(
            _grad_data(out), max_indices, batches=1, image_size=image.numel,
        ))

    _set_backward(out, _backward)
    return out


def _pool2d_setup(
    name: str,
    image: Tensor,
    window: tuple[int, int],
    stride: tuple[int, int] | None,
) -> tuple[int, tuple[int, ...], tuple[int, int]]:
    """Validate pooling arguments; return (planes, out shape, stride)."""

    if image.ndim not in (2, 3, 4):
        raise ValueError(f"{name} expects a 2D, 3D, or 4D tensor")

    image_rows, image_cols = image.shape[-2:]
    window_rows, window_cols = window
    if window_rows <= 0 or window_cols <= 0:
        raise ValueError(f"{name} window dimensions must be positive")

    if stride is None:
        stride_rows, stride_cols = window
    else:
        stride_rows, stride_cols = stride
        if stride_rows <= 0 or stride_cols <= 0:
            raise ValueError(f"{name} stride dimensions must be positive")

    if window_rows > image_rows or window_cols > image_cols:
        raise ValueError(f"{name} window must fit inside the image")

    out_rows = ((image_rows - window_rows) // stride_rows) + 1
    out_cols = ((image_cols - window_cols) // stride_cols) + 1
    planes = _numel(image.shape[:-2])
    out_shape = (*image.shape[:-2], out_rows, out_cols)
    return planes, out_shape, (stride_rows, stride_cols)


@lru_cache(maxsize=64)
def _pool2d_window_index(
    image_shape: tuple[int, int],
    window: tuple[int, int],
    stride: tuple[int, int],
) -> tuple[int, ...]:
    """Flat plane offsets of every window's pixels, window after window."""

    image_rows, image_cols = image_shape
    window_rows, window_cols = window
    stride_rows, stride_cols = stride
    return kernels.index_table(
        (out_row * stride_rows + row) * image_cols + out_col * stride_cols + col
        for out_row in range((image_rows - window_rows) // stride_rows + 1)
        for out_col in range((image_cols - window_cols) // stride_cols + 1)
        for row in range(window_rows)
        for col in range(window_cols)
    )


# ---------------------------------------------------------------------------
# Small index helpers
# ---------------------------------------------------------------------------


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Row-major strides: how far to step in flat data per axis step."""

    strides = []
    step = 1
    for dim in reversed(shape):
        strides.append(step)
        step *= dim
    return tuple(reversed(strides))


def _strided_indices(
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
) -> tuple[int, ...]:
    """Flat offsets visited by walking ``sizes`` in row-major order.

    Each axis multiplies the list: every existing offset is extended by every
    step along the new axis.
    """

    indices = [0]
    for size, stride in zip(sizes, strides):
        steps = [step * stride for step in range(size)]
        indices = [base + step for base in indices for step in steps]
    return tuple(indices)


def _flat_index(index: tuple[int, ...], shape: tuple[int, ...]) -> int:
    offset = 0
    for value, dim in zip(index, shape):
        offset = offset * dim + _normalize_index(value, dim)
    return offset


def _unravel_index(flat_index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    index = []
    for dim in reversed(shape):
        index.append(flat_index % dim)
        flat_index //= dim
    return tuple(reversed(index))


def _normalize_axis(axis: int, ndim: int) -> int:
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError("axis out of range")
    return axis


def _normalize_insert_axis(axis: int, ndim: int) -> int:
    if axis < 0:
        axis += ndim + 1
    if axis < 0 or axis > ndim:
        raise ValueError("axis out of range")
    return axis


def _normalize_axes(axes: tuple[int, ...], ndim: int) -> tuple[int, ...]:
    if len(axes) != ndim:
        raise ValueError("axes length must match tensor dimensions")

    normalized = tuple(_normalize_axis(axis, ndim) for axis in axes)
    if sorted(normalized) != list(range(ndim)):
        raise ValueError("axes must be a permutation of tensor dimensions")
    return normalized


def _reduced_shape(shape: tuple[int, ...], axis: int) -> tuple[int, ...]:
    reduced = shape[:axis] + shape[axis + 1 :]
    if not reduced:
        return (1,)
    return reduced


def _infer_shape(values: Sequence) -> tuple[int, ...]:
    if not _is_sequence(values):
        raise ValueError("values must be a non-empty sequence")
    if not values:
        raise ValueError("values must not be empty")

    first = values[0]
    if not _is_sequence(first):
        if any(_is_sequence(value) for value in values):
            raise ValueError("nested input must be rectangular")
        return (len(values),)

    child_shape = _infer_shape(first)
    for value in values:
        if not _is_sequence(value) or _infer_shape(value) != child_shape:
            raise ValueError("nested input must be rectangular")
    return (len(values), *child_shape)


def _flatten_nested(values: Sequence) -> list[float]:
    flat = []
    for value in values:
        if _is_sequence(value):
            flat.extend(_flatten_nested(value))
        else:
            flat.append(float(value))
    return flat


def _nested_list(data: list[float], shape: tuple[int, ...]) -> list:
    if len(shape) == 1:
        return list(data[: shape[0]])

    step = _numel(shape[1:])
    return [
        _nested_list(data[index * step : (index + 1) * step], shape[1:])
        for index in range(shape[0])
    ]


def _is_sequence(value) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _normalize_index(index: int, size: int) -> int:
    if index < 0:
        index += size
    if index < 0 or index >= size:
        raise IndexError("tensor index out of range")
    return index
