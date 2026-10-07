"""Small tensor container, shape-checked math helpers, and tensor autograd.

Everything here is plain Python lists of floats in row-major order. The file
reads top to bottom as:

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
- Every matrix product is a table of dot products computed by one helper,
  ``_dot_table``. Backward passes reuse the same helper:
  ``dA = dC @ B^T`` and ``dB = A^T @ dC``.
- Convolution is "im2col": gather every image patch into a row, then do one
  dot-product table against the flattened kernels.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import lru_cache
from operator import mul
from typing import Callable, NamedTuple


Number = int | float


# ---------------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------------


class Tensor:
    """A dense tensor backed by flat row-major data."""

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

        self.data = [float(value) for value in data]
        self.shape = tuple(shape)
        self.requires_grad = requires_grad
        self.grad: list[float] | None = (
            [0.0] * len(self.data)
            if requires_grad
            else None
        )

        self._prev = set(_children)
        self._op = _op
        self._backward: Callable[[], None] = _no_backward

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

    def __getitem__(self, index: int | tuple[int, ...]) -> float:
        if len(self.shape) == 1:
            if not isinstance(index, int):
                raise IndexError("1D tensors use one integer index")
            return self.data[_normalize_index(index, self.shape[0])]

        if not isinstance(index, tuple) or len(index) != len(self.shape):
            raise IndexError("tensor indices must match tensor dimensions")

        return self.data[_flat_index(index, self.shape)]

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

    def permute(self, axes: tuple[int, ...]) -> "Tensor":
        return permute(self, axes)

    def conv2d_valid(self, kernel: "Tensor") -> "Tensor":
        return conv2d_valid(self, kernel)

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
            self.grad = [0.0] * len(self.data)

    def backward(self, grad: Tensor | Sequence[Number] | None = None) -> None:
        """Fill ``.grad`` on every tensor this one was computed from.

        Gradients are reset before the pass, so calling backward twice gives
        the same result rather than accumulating.
        """

        topo = _topological_order(self)

        if self.grad is None:
            self.grad = [0.0] * len(self.data)
        for node in topo:
            if not node.requires_grad:
                continue
            if node._prev:
                # Intermediate results start empty; _add_grad fills them.
                node.grad = None
            else:
                node.zero_grad()

        _add_grad(self, _seed_grad(self, grad))

        for node in reversed(topo):
            if node.requires_grad:
                node._backward()


def _no_backward() -> None:
    return None


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
    pass trusted float lists, so no validation or copy happens here.

    A custom op is three steps: compute ``data``, call ``_op_result``, then
    set ``out._backward`` to a function that reads ``out.grad`` and calls
    ``_add_grad`` on each parent that requires grad.
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
    return out


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
    return [float(value) for value in grad]


def _grad_data(tensor: Tensor) -> list[float]:
    if tensor.grad is None:
        return [0.0] * len(tensor.data)
    return tensor.grad


def _add_grad(tensor: Tensor, grad: Sequence[Number]) -> None:
    """Accumulate ``grad`` into ``tensor.grad`` (gradients add up over uses)."""

    if len(grad) != len(tensor.data):
        raise ValueError("gradient length does not match tensor data")
    if tensor.grad is None:
        tensor.grad = [float(value) for value in grad]
        return
    tensor.grad = [
        current + value
        for current, value in zip(tensor.grad, grad)
    ]


# ---------------------------------------------------------------------------
# Elementwise math with broadcasting
# ---------------------------------------------------------------------------


def add(left: Tensor, right: Number | Tensor) -> Tensor:
    """Elementwise ``left + right`` with numpy-style broadcasting.

    d(out)/d(left) = 1 and d(out)/d(right) = 1, so both sides receive the
    output gradient, summed back over any broadcast axes.
    """

    if isinstance(right, (int, float)):
        out = _op_result(
            [value + right for value in left.data],
            left.shape,
            (left,),
            "+",
        )

        def _backward_scalar() -> None:
            _add_grad(left, _grad_data(out))

        out._backward = _backward_scalar
        return out

    out_shape = _broadcast_shape(left.shape, right.shape)
    left_values = _expand(left, out_shape)
    right_values = _expand(right, out_shape)
    out = _op_result(
        [a + b for a, b in zip(left_values, right_values)],
        out_shape,
        (left, right),
        "+",
    )

    def _backward() -> None:
        grad = _grad_data(out)
        if left.requires_grad:
            _add_grad(left, _reduce_to_shape(grad, out_shape, left.shape))
        if right.requires_grad:
            _add_grad(right, _reduce_to_shape(grad, out_shape, right.shape))

    out._backward = _backward
    return out


def subtract(left: Tensor, right: Tensor) -> Tensor:
    """Elementwise ``left - right`` with numpy-style broadcasting."""

    out_shape = _broadcast_shape(left.shape, right.shape)
    left_values = _expand(left, out_shape)
    right_values = _expand(right, out_shape)
    out = _op_result(
        [a - b for a, b in zip(left_values, right_values)],
        out_shape,
        (left, right),
        "-",
    )

    def _backward() -> None:
        grad = _grad_data(out)
        if left.requires_grad:
            _add_grad(left, _reduce_to_shape(grad, out_shape, left.shape))
        if right.requires_grad:
            negated = [-value for value in grad]
            _add_grad(right, _reduce_to_shape(negated, out_shape, right.shape))

    out._backward = _backward
    return out


def multiply(left: Tensor, right: Number | Tensor) -> Tensor:
    """Elementwise ``left * right`` with numpy-style broadcasting.

    d(out)/d(left) = right and d(out)/d(right) = left.
    """

    if isinstance(right, (int, float)):
        out = _op_result(
            [value * right for value in left.data],
            left.shape,
            (left,),
            "*",
        )

        def _backward_scalar() -> None:
            _add_grad(left, [right * grad for grad in _grad_data(out)])

        out._backward = _backward_scalar
        return out

    out_shape = _broadcast_shape(left.shape, right.shape)
    left_values = _expand(left, out_shape)
    right_values = _expand(right, out_shape)
    out = _op_result(
        [a * b for a, b in zip(left_values, right_values)],
        out_shape,
        (left, right),
        "*",
    )

    def _backward() -> None:
        grad = _grad_data(out)
        if left.requires_grad:
            left_grad = [g * b for g, b in zip(grad, right_values)]
            _add_grad(left, _reduce_to_shape(left_grad, out_shape, left.shape))
        if right.requires_grad:
            right_grad = [g * a for g, a in zip(grad, left_values)]
            _add_grad(right, _reduce_to_shape(right_grad, out_shape, right.shape))

    out._backward = _backward
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


def _expand(tensor: Tensor, out_shape: tuple[int, ...]) -> list[float]:
    """Values of ``tensor`` laid out over ``out_shape`` (no copy if equal)."""

    if tensor.shape == out_shape:
        return tensor.data
    data = tensor.data
    return [data[index] for index in _broadcast_index_map(out_shape, tensor.shape)]


def _reduce_to_shape(
    grad: list[float],
    out_shape: tuple[int, ...],
    source_shape: tuple[int, ...],
) -> list[float]:
    """Sum a broadcast gradient back down to the source tensor's shape."""

    if out_shape == source_shape:
        return grad
    source_grad = [0.0] * _numel(source_shape)
    for index, value in zip(_broadcast_index_map(out_shape, source_shape), grad):
        source_grad[index] += value
    return source_grad


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

    out._backward = _backward
    return out


def tensor_exp(tensor: Tensor) -> Tensor:
    """Elementwise exponential: y = e^x, dy/dx = y."""

    out_data = [math.exp(value) for value in tensor.data]
    return _unary(
        tensor,
        out_data,
        lambda grad: [g * y for g, y in zip(grad, out_data)],
        "exp",
    )


def tensor_log(tensor: Tensor) -> Tensor:
    """Elementwise natural logarithm: y = ln x, dy/dx = 1 / x."""

    if any(value <= 0 for value in tensor.data):
        raise ValueError("log is only defined for positive values")
    inputs = tensor.data
    return _unary(
        tensor,
        [math.log(value) for value in inputs],
        lambda grad: [g / x for g, x in zip(grad, inputs)],
        "log",
    )


def tensor_relu(tensor: Tensor) -> Tensor:
    """Elementwise rectified linear unit: y = max(0, x)."""

    inputs = tensor.data
    return _unary(
        tensor,
        [value if value > 0.0 else 0.0 for value in inputs],
        lambda grad: [g if x > 0.0 else 0.0 for g, x in zip(grad, inputs)],
        "relu",
    )


def tensor_gelu(tensor: Tensor) -> Tensor:
    """Elementwise Gaussian error linear unit (exact erf form).

    y  = x * Phi(x)                 Phi = standard normal CDF
    dy = Phi(x) + x * phi(x)        phi = standard normal PDF
    """

    sqrt_two = math.sqrt(2.0)
    sqrt_two_pi = math.sqrt(2.0 * math.pi)
    inputs = tensor.data
    cdf = [0.5 * (1.0 + math.erf(x / sqrt_two)) for x in inputs]

    def input_grad(grad: list[float]) -> list[float]:
        return [
            g * (phi + x * math.exp(-0.5 * x * x) / sqrt_two_pi)
            for g, x, phi in zip(grad, inputs, cdf)
        ]

    return _unary(
        tensor,
        [x * phi for x, phi in zip(inputs, cdf)],
        input_grad,
        "gelu",
    )


def tensor_sigmoid(tensor: Tensor) -> Tensor:
    """Elementwise sigmoid: y = 1 / (1 + e^-x), dy/dx = y (1 - y)."""

    out_data = [_sigmoid(value) for value in tensor.data]
    return _unary(
        tensor,
        out_data,
        lambda grad: [g * y * (1.0 - y) for g, y in zip(grad, out_data)],
        "sigmoid",
    )


def _sigmoid(value: float) -> float:
    # Two algebraically equal forms; each avoids overflow on its own side.
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def tensor_tanh(tensor: Tensor) -> Tensor:
    """Elementwise hyperbolic tangent: dy/dx = 1 - y^2."""

    out_data = [math.tanh(value) for value in tensor.data]
    return _unary(
        tensor,
        out_data,
        lambda grad: [g * (1.0 - y * y) for g, y in zip(grad, out_data)],
        "tanh",
    )


def tensor_reciprocal(tensor: Tensor) -> Tensor:
    """Elementwise reciprocal: y = 1 / x, dy/dx = -y^2."""

    if any(value == 0.0 for value in tensor.data):
        raise ValueError("division by zero")
    out_data = [1.0 / value for value in tensor.data]
    return _unary(
        tensor,
        out_data,
        lambda grad: [-g * y * y for g, y in zip(grad, out_data)],
        "reciprocal",
    )


# ---------------------------------------------------------------------------
# Matrix multiplication
# ---------------------------------------------------------------------------


def _python_dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(map(mul, left, right))


# math.sumprod (Python 3.12+) is a C-speed dot product.
_dot: Callable[[Sequence[float], Sequence[float]], float] = getattr(
    math,
    "sumprod",
    _python_dot,
)


def _rows(data: list[float], width: int) -> list[list[float]]:
    """Split a flat row-major matrix into its rows."""

    return [data[start : start + width] for start in range(0, len(data), width)]


def _columns(data: list[float], width: int) -> list[list[float]]:
    """Split a flat row-major matrix into its columns."""

    return [data[col::width] for col in range(width)]


def _dot_table(
    row_vectors: Sequence[Sequence[float]],
    column_vectors: Sequence[Sequence[float]],
) -> list[float]:
    """Flat (rows, cols) table with ``out[i, j] = row_i . column_j``.

    This is the whole matrix-multiply kernel: ``A @ B`` is the dot table of
    the rows of A against the columns of B.
    """

    out: list[float] = []
    for row in row_vectors:
        out.extend([_dot(row, column) for column in column_vectors])
    return out


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
    batches, rows, shared, cols = plan.batches, plan.rows, plan.shared, plan.cols
    left_step, right_step = plan.left_step, plan.right_step
    left_size = rows * shared
    right_size = shared * cols
    out_size = rows * cols
    left_data = left.data
    right_data = right.data

    data: list[float] = []
    for batch in range(batches):
        a = left_data[batch * left_step : batch * left_step + left_size]
        b = right_data[batch * right_step : batch * right_step + right_size]
        data.extend(_dot_table(_rows(a, shared), _columns(b, cols)))

    out = _op_result(data, plan.out_shape, (left, right), "matmul")

    def _backward() -> None:
        grad = _grad_data(out)
        left_grad = [0.0] * left.numel if left.requires_grad else None
        right_grad = [0.0] * right.numel if right.requires_grad else None

        for batch in range(batches):
            a_start = batch * left_step
            b_start = batch * right_step
            a = left_data[a_start : a_start + left_size]
            b = right_data[b_start : b_start + right_size]
            d_c = grad[batch * out_size : (batch + 1) * out_size]

            if left_grad is not None:
                # dA = dC @ B^T: rows of dC dotted with rows of B.
                d_a = _dot_table(_rows(d_c, cols), _rows(b, cols))
                _accumulate(left_grad, a_start, d_a)
            if right_grad is not None:
                # dB = A^T @ dC: columns of A dotted with columns of dC.
                d_b = _dot_table(_columns(a, shared), _columns(d_c, cols))
                _accumulate(right_grad, b_start, d_b)

        if left_grad is not None:
            _add_grad(left, left_grad)
        if right_grad is not None:
            _add_grad(right, right_grad)

    out._backward = _backward
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


def _accumulate(target: list[float], start: int, values: list[float]) -> None:
    stop = start + len(values)
    target[start:stop] = [
        current + value
        for current, value in zip(target[start:stop], values)
    ]


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

    out._backward = _backward
    return out


def flatten(tensor: Tensor, start_axis: int = 0) -> Tensor:
    """Return tensor values flattened from start_axis onward."""

    start_axis = _normalize_axis(start_axis, tensor.ndim)
    leading_shape = tensor.shape[:start_axis]
    flattened = _numel(tensor.shape[start_axis:])
    return reshape(tensor, (*leading_shape, flattened))


def permute(tensor: Tensor, axes: tuple[int, ...]) -> Tensor:
    """Return a tensor with axes reordered: out axis i is input axis axes[i]."""

    axes = _normalize_axes(tuple(axes), tensor.ndim)
    out_shape = tuple(tensor.shape[axis] for axis in axes)
    source_indices = _permute_index_map(tensor.shape, axes)
    data = tensor.data
    out = _op_result(
        [data[index] for index in source_indices],
        out_shape,
        (tensor,),
        "permute",
    )

    def _backward() -> None:
        # A permutation moves every value exactly once, so just move back.
        tensor_grad = [0.0] * tensor.numel
        for index, value in zip(source_indices, _grad_data(out)):
            tensor_grad[index] = value
        _add_grad(tensor, tensor_grad)

    out._backward = _backward
    return out


@lru_cache(maxsize=256)
def _permute_index_map(
    shape: tuple[int, ...],
    axes: tuple[int, ...],
) -> tuple[int, ...]:
    """For each flat output position, the flat input position it reads."""

    strides = _strides(shape)
    return _strided_indices(
        tuple(shape[axis] for axis in axes),
        tuple(strides[axis] for axis in axes),
    )


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

    outer = len(tensors[0].data) // chunks[0]
    data: list[float] = []
    for outer_index in range(outer):
        for tensor, chunk in zip(tensors, chunks):
            start = outer_index * chunk
            data.extend(tensor.data[start : start + chunk])

    out = _op_result(data, out_shape, tuple(tensors), op)

    def _backward() -> None:
        grad = _grad_data(out)
        tensor_grads: list[list[float]] = [[] for _ in tensors]
        offset = 0
        for _ in range(outer):
            for tensor_grad, chunk in zip(tensor_grads, chunks):
                tensor_grad.extend(grad[offset : offset + chunk])
                offset += chunk
        for tensor, tensor_grad in zip(tensors, tensor_grads):
            if tensor.requires_grad:
                _add_grad(tensor, tensor_grad)

    out._backward = _backward
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
    for index in indices:
        if index < 0 or index >= rows:
            raise ValueError("embedding index out of range")

    table_rows = _rows(table.data, width)
    data: list[float] = []
    for index in indices:
        data.extend(table_rows[index])

    out = _op_result(data, (len(indices), width), (table,), "gather_rows")

    def _backward() -> None:
        grad = _grad_data(out)
        table_grad = [0.0] * table.numel
        for position, index in enumerate(indices):
            _accumulate(
                table_grad,
                index * width,
                grad[position * width : (position + 1) * width],
            )
        _add_grad(table, table_grad)

    out._backward = _backward
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

        out._backward = _backward_all
        return out

    axis = _normalize_axis(axis, len(tensor.shape))
    lanes = _axis_lanes(tensor.shape, axis)
    size = tensor.shape[axis]
    data = tensor.data
    out = _op_result(
        [sum(data[lane]) for lane in lanes],
        _reduced_shape(tensor.shape, axis),
        (tensor,),
        "sum",
    )

    def _backward() -> None:
        # Every value in a lane contributed once to that lane's sum.
        tensor_grad = [0.0] * tensor.numel
        for lane, value in zip(lanes, _grad_data(out)):
            tensor_grad[lane] = [value] * size
        _add_grad(tensor, tensor_grad)

    out._backward = _backward
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

    out._backward = _backward
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


def conv2d_valid(image: Tensor, kernel: Tensor) -> Tensor:
    """Apply a valid (no padding, stride 1) convolution-style filter.

    This uses the cross-correlation convention common in neural networks:
    the kernel is slid over the image without flipping it.

    Accepted shapes, with B batch, C input channels, O output filters:

    - image (H, W)       kernel (kh, kw)       -> (oh, ow)
    - image (H, W)       kernel (O, kh, kw)    -> (O, oh, ow)
    - image (C, H, W)    kernel (C, kh, kw)    -> (oh, ow)
    - image (C, H, W)    kernel (O, C, kh, kw) -> (O, oh, ow)

    Unbatched outputs with a single filter (O = 1) drop the filter axis.
    - image (B, C, H, W) kernel (O, C, kh, kw) -> (B, O, oh, ow)

    Internally everything is treated as the batched 4D case (reshaping a flat
    row-major list is free), using im2col:

        patches[p] = the C*kh*kw image values under the kernel at position p
        out[b, o, p] = kernel_o . patches[p]
    """

    batches, channels, image_rows, image_cols, filters, out_shape = (
        _conv2d_shapes(image.shape, kernel.shape)
    )
    kernel_rows, kernel_cols = kernel.shape[-2:]
    patch_indices = _conv2d_patch_indices(
        channels,
        image_rows,
        image_cols,
        kernel_rows,
        kernel_cols,
    )
    patch_size = channels * kernel_rows * kernel_cols
    image_size = channels * image_rows * image_cols
    positions = len(patch_indices)
    kernel_vectors = _rows(kernel.data, patch_size)

    image_data = image.data
    batch_patches: list[list[list[float]]] = []
    data: list[float] = []
    for batch in range(batches):
        single = image_data[batch * image_size : (batch + 1) * image_size]
        patches = [
            [single[index] for index in indices]
            for indices in patch_indices
        ]
        batch_patches.append(patches)
        data.extend(_dot_table(kernel_vectors, patches))

    out = _op_result(data, out_shape, (image, kernel), "conv2d_valid")

    def _backward() -> None:
        grad = _grad_data(out)
        kernel_grad = [0.0] * kernel.numel if kernel.requires_grad else None
        image_grad = [0.0] * image.numel if image.requires_grad else None

        for batch, patches in enumerate(batch_patches):
            out_start = batch * filters * positions
            d_out = grad[out_start : out_start + filters * positions]

            if kernel_grad is not None:
                # dK = dOut @ patches: (O, P) @ (P, C*kh*kw)
                patch_columns = list(zip(*patches))
                d_kernel = _dot_table(_rows(d_out, positions), patch_columns)
                _accumulate(kernel_grad, 0, d_kernel)

            if image_grad is not None:
                # dPatches = dOut^T @ K: (P, O) @ (O, C*kh*kw), then scatter
                # each patch gradient back onto the pixels it was read from.
                d_patches = _dot_table(
                    _columns(d_out, positions),
                    _columns(kernel.data, patch_size),
                )
                image_start = batch * image_size
                for position, indices in enumerate(patch_indices):
                    row_start = position * patch_size
                    for offset, index in enumerate(indices):
                        image_grad[image_start + index] += d_patches[row_start + offset]

        if kernel_grad is not None:
            _add_grad(kernel, kernel_grad)
        if image_grad is not None:
            _add_grad(image, image_grad)

    out._backward = _backward
    return out


def _conv2d_shapes(
    image_shape: tuple[int, ...],
    kernel_shape: tuple[int, ...],
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

    out_rows = image_rows - kernel_rows + 1
    out_cols = image_cols - kernel_cols + 1
    if len(image_shape) == 4:
        out_shape = (batches, filters, out_rows, out_cols)
    elif filters == 1:
        # Unbatched single-filter output drops the filter axis.
        out_shape = (out_rows, out_cols)
    else:
        out_shape = (filters, out_rows, out_cols)
    return batches, channels, image_rows, image_cols, filters, out_shape


@lru_cache(maxsize=64)
def _conv2d_patch_indices(
    channels: int,
    image_rows: int,
    image_cols: int,
    kernel_rows: int,
    kernel_cols: int,
) -> tuple[tuple[int, ...], ...]:
    """For every output position, the flat image indices under the kernel.

    Index order inside a patch matches a flattened (C, kh, kw) kernel.
    """

    channel_size = image_rows * image_cols
    patch_offsets = [
        channel * channel_size + row * image_cols + col
        for channel in range(channels)
        for row in range(kernel_rows)
        for col in range(kernel_cols)
    ]
    return tuple(
        tuple(out_row * image_cols + out_col + offset for offset in patch_offsets)
        for out_row in range(image_rows - kernel_rows + 1)
        for out_col in range(image_cols - kernel_cols + 1)
    )


def avg_pool2d(
    image: Tensor,
    window: tuple[int, int],
    *,
    stride: tuple[int, int] | None = None,
) -> Tensor:
    """Average-pool a 2D tensor or each channel in a 3D/4D tensor."""

    planes, out_shape, windows = _pool2d_setup("avg_pool2d", image, window, stride)
    plane_size = image.shape[-2] * image.shape[-1]
    scale = 1 / (window[0] * window[1])
    source = image.data

    data: list[float] = []
    for plane in range(planes):
        offset = plane * plane_size
        data.extend([
            sum(source[offset + index] for index in indices) * scale
            for indices in windows
        ])

    out = _op_result(data, out_shape, (image,), "avg_pool2d")

    def _backward() -> None:
        # Every pixel in a window gets an equal share of that window's grad.
        grad = _grad_data(out)
        image_grad = [0.0] * image.numel
        grad_index = 0
        for plane in range(planes):
            offset = plane * plane_size
            for indices in windows:
                share = grad[grad_index] * scale
                grad_index += 1
                for index in indices:
                    image_grad[offset + index] += share
        _add_grad(image, image_grad)

    out._backward = _backward
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

    planes, out_shape, windows = _pool2d_setup("max_pool2d", image, window, stride)
    plane_size = image.shape[-2] * image.shape[-1]
    source = image.data

    data: list[float] = []
    max_indices: list[int] = []
    for plane in range(planes):
        offset = plane * plane_size
        for indices in windows:
            best_index = offset + indices[0]
            for index in indices[1:]:
                if source[offset + index] > source[best_index]:
                    best_index = offset + index
            data.append(source[best_index])
            max_indices.append(best_index)

    out = _op_result(data, out_shape, (image,), "max_pool2d")

    def _backward() -> None:
        # Only the winning pixel of each window affected the output.
        image_grad = [0.0] * image.numel
        for out_grad, image_index in zip(_grad_data(out), max_indices):
            image_grad[image_index] += out_grad
        _add_grad(image, image_grad)

    out._backward = _backward
    return out


def _pool2d_setup(
    name: str,
    image: Tensor,
    window: tuple[int, int],
    stride: tuple[int, int] | None,
) -> tuple[int, tuple[int, ...], tuple[tuple[int, ...], ...]]:
    """Validate pooling arguments; return (planes, out shape, windows)."""

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
    windows = _pool2d_windows(
        image_cols,
        window_rows,
        window_cols,
        stride_rows,
        stride_cols,
        out_rows,
        out_cols,
    )
    return planes, out_shape, windows


@lru_cache(maxsize=64)
def _pool2d_windows(
    image_cols: int,
    window_rows: int,
    window_cols: int,
    stride_rows: int,
    stride_cols: int,
    out_rows: int,
    out_cols: int,
) -> tuple[tuple[int, ...], ...]:
    """For every output position, the flat indices of its window in a plane."""

    return tuple(
        tuple(
            (out_row * stride_rows + row) * image_cols + out_col * stride_cols + col
            for row in range(window_rows)
            for col in range(window_cols)
        )
        for out_row in range(out_rows)
        for out_col in range(out_cols)
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
        return data[: shape[0]]

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
