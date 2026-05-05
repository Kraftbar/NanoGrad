"""Small tensor container, shape-checked math helpers, and tensor autograd."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Callable


Number = int | float


class Tensor:
    """A dense 1D or 2D tensor backed by flat row-major data."""

    def __init__(
        self,
        data: Sequence[Number],
        shape: tuple[int, ...],
        *,
        requires_grad: bool = False,
        _children: tuple["Tensor", ...] = (),
        _op: str = "",
    ) -> None:
        if len(shape) not in (1, 2):
            raise ValueError("Tensor currently supports only 1D or 2D shapes")
        if any(dim <= 0 for dim in shape):
            raise ValueError("shape dimensions must be positive")
        if len(data) != _numel(shape):
            raise ValueError("data length does not match shape")

        self.data = [float(value) for value in data]
        self.shape = shape
        self.requires_grad = requires_grad
        self.grad: list[float] | None = (
            [0.0] * len(self.data)
            if requires_grad
            else None
        )

        self._prev = set(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, data={self.tolist()})"

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index: int | tuple[int, int]) -> float:
        if len(self.shape) == 1:
            if not isinstance(index, int):
                raise IndexError("1D tensors use one integer index")
            return self.data[_normalize_index(index, self.shape[0])]

        if not isinstance(index, tuple) or len(index) != 2:
            raise IndexError("2D tensors use row and column indices")

        row = _normalize_index(index[0], self.shape[0])
        col = _normalize_index(index[1], self.shape[1])
        return self.data[row * self.shape[1] + col]

    def tolist(self) -> list[float] | list[list[float]]:
        if len(self.shape) == 1:
            return self.data[:]

        rows, cols = self.shape
        return [
            self.data[row * cols : (row + 1) * cols]
            for row in range(rows)
        ]

    @classmethod
    def zeros(cls, shape: tuple[int, ...], *, requires_grad: bool = False) -> "Tensor":
        return cls([0.0] * _numel(shape), shape, requires_grad=requires_grad)

    @classmethod
    def from_list(
        cls,
        values: Sequence[Number] | Sequence[Sequence[Number]],
        *,
        requires_grad: bool = False,
    ) -> "Tensor":
        if not values:
            raise ValueError("values must not be empty")

        first = values[0]
        if isinstance(first, Sequence):
            rows = values
            width = len(first)
            if width == 0:
                raise ValueError("rows must not be empty")
            if any(not isinstance(row, Sequence) or len(row) != width for row in rows):
                raise ValueError("2D input must be rectangular")

            data = [
                value
                for row in rows
                for value in row
            ]
            return cls(data, (len(rows), width), requires_grad=requires_grad)

        return cls(values, (len(values),), requires_grad=requires_grad)

    def __add__(self, other: Number | "Tensor") -> "Tensor":
        return add(self, other)

    def __radd__(self, other: Number | "Tensor") -> "Tensor":
        return add(self, other)

    def __sub__(self, other: Number | "Tensor") -> "Tensor":
        return add(self, -other if isinstance(other, (int, float)) else other * -1)

    def __rsub__(self, other: Number | "Tensor") -> "Tensor":
        return add(-self, other)

    def __neg__(self) -> "Tensor":
        return self * -1

    def __mul__(self, other: Number | "Tensor") -> "Tensor":
        return multiply(self, other)

    def __rmul__(self, other: Number | "Tensor") -> "Tensor":
        return multiply(self, other)

    @property
    def T(self) -> "Tensor":
        return transpose(self)

    def sum(self, axis: int | None = None) -> "Tensor":
        return tensor_sum(self, axis=axis)

    def mean(self, axis: int | None = None) -> "Tensor":
        return tensor_mean(self, axis=axis)

    def exp(self) -> "Tensor":
        return tensor_exp(self)

    def log(self) -> "Tensor":
        return tensor_log(self)

    def relu(self) -> "Tensor":
        return tensor_relu(self)

    def sigmoid(self) -> "Tensor":
        return tensor_sigmoid(self)

    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad = [0.0] * len(self.data)

    def backward(self, grad: Tensor | Sequence[Number] | None = None) -> None:
        topo: list[Tensor] = []
        visited: set[Tensor] = set()

        def build(node: Tensor) -> None:
            if node in visited:
                return
            visited.add(node)
            for child in node._prev:
                build(child)
            topo.append(node)

        build(self)

        if self.grad is None:
            self.grad = [0.0] * len(self.data)
        for node in topo:
            if node.requires_grad:
                node.zero_grad()

        seed = _seed_grad(self, grad)
        for i, value in enumerate(seed):
            self.grad[i] += value

        for node in reversed(topo):
            node._backward()


def add(left: Tensor, right: Number | Tensor) -> Tensor:
    """Elementwise addition with scalar and row-vector broadcasting."""

    if isinstance(right, (int, float)):
        out = Tensor(
            [value + right for value in left.data],
            left.shape,
            requires_grad=left.requires_grad,
            _children=(left,),
            _op="+",
        )

        def _backward() -> None:
            if left.requires_grad:
                _add_grad(left, _grad_data(out))

        out._backward = _backward
        return out

    if _can_row_broadcast(left, right):
        out = _row_broadcast(left, right, lambda a, b: a + b, _op="+")

        def _backward() -> None:
            grad = _grad_data(out)
            if left.requires_grad:
                _add_grad(left, grad)
            if right.requires_grad:
                _add_grad(right, _sum_row_broadcast_grad(left.shape, grad))

        out._backward = _backward
        return out

    if _can_row_broadcast(right, left):
        out = _row_broadcast(right, left, lambda a, b: b + a, _op="+")

        def _backward() -> None:
            grad = _grad_data(out)
            if right.requires_grad:
                _add_grad(right, grad)
            if left.requires_grad:
                _add_grad(left, _sum_row_broadcast_grad(right.shape, grad))

        out._backward = _backward
        return out

    _require_same_shape(left, right)
    out = Tensor(
        [
            a + b
            for a, b in zip(left.data, right.data)
        ],
        left.shape,
        requires_grad=left.requires_grad or right.requires_grad,
        _children=(left, right),
        _op="+",
    )

    def _backward() -> None:
        grad = _grad_data(out)
        if left.requires_grad:
            _add_grad(left, grad)
        if right.requires_grad:
            _add_grad(right, grad)

    out._backward = _backward
    return out


def multiply(left: Tensor, right: Number | Tensor) -> Tensor:
    """Elementwise multiplication with scalar and row-vector broadcasting."""

    if isinstance(right, (int, float)):
        out = Tensor(
            [value * right for value in left.data],
            left.shape,
            requires_grad=left.requires_grad,
            _children=(left,),
            _op="*",
        )

        def _backward() -> None:
            if left.requires_grad:
                _add_grad(left, [right * grad for grad in _grad_data(out)])

        out._backward = _backward
        return out

    if _can_row_broadcast(left, right):
        out = _row_broadcast(left, right, lambda a, b: a * b, _op="*")

        def _backward() -> None:
            grad = _grad_data(out)
            rows, cols = left.shape
            if left.requires_grad:
                _add_grad(
                    left,
                    [
                        grad[row * cols + col] * right[col]
                        for row in range(rows)
                        for col in range(cols)
                    ],
                )
            if right.requires_grad:
                row_grad = [0.0] * cols
                for row in range(rows):
                    for col in range(cols):
                        row_grad[col] += grad[row * cols + col] * left[row, col]
                _add_grad(right, row_grad)

        out._backward = _backward
        return out

    if _can_row_broadcast(right, left):
        out = _row_broadcast(right, left, lambda a, b: b * a, _op="*")

        def _backward() -> None:
            grad = _grad_data(out)
            rows, cols = right.shape
            if right.requires_grad:
                _add_grad(
                    right,
                    [
                        grad[row * cols + col] * left[col]
                        for row in range(rows)
                        for col in range(cols)
                    ],
                )
            if left.requires_grad:
                row_grad = [0.0] * cols
                for row in range(rows):
                    for col in range(cols):
                        row_grad[col] += grad[row * cols + col] * right[row, col]
                _add_grad(left, row_grad)

        out._backward = _backward
        return out

    _require_same_shape(left, right)
    out = Tensor(
        [
            a * b
            for a, b in zip(left.data, right.data)
        ],
        left.shape,
        requires_grad=left.requires_grad or right.requires_grad,
        _children=(left, right),
        _op="*",
    )

    def _backward() -> None:
        grad = _grad_data(out)
        if left.requires_grad:
            _add_grad(left, [b * g for b, g in zip(right.data, grad)])
        if right.requires_grad:
            _add_grad(right, [a * g for a, g in zip(left.data, grad)])

    out._backward = _backward
    return out


def matmul(left: Tensor, right: Tensor) -> Tensor:
    """Matrix/vector multiplication for 1D and 2D tensors."""

    if len(left.shape) == 1 and len(right.shape) == 1:
        if left.shape[0] != right.shape[0]:
            raise ValueError("vector dot product shapes must match")
        total = sum(
            a * b
            for a, b in zip(left.data, right.data)
        )
        out = Tensor(
            [total],
            (1,),
            requires_grad=left.requires_grad or right.requires_grad,
            _children=(left, right),
            _op="matmul",
        )

        def _backward() -> None:
            grad = _grad_data(out)[0]
            if left.requires_grad:
                _add_grad(left, [value * grad for value in right.data])
            if right.requires_grad:
                _add_grad(right, [value * grad for value in left.data])

        out._backward = _backward
        return out

    if len(left.shape) == 2 and len(right.shape) == 1:
        rows, shared = left.shape
        if shared != right.shape[0]:
            raise ValueError("matrix-vector inner dimensions must match")

        data = []
        for row in range(rows):
            total = 0.0
            for col in range(shared):
                total += left[row, col] * right[col]
            data.append(total)
        out = Tensor(
            data,
            (rows,),
            requires_grad=left.requires_grad or right.requires_grad,
            _children=(left, right),
            _op="matmul",
        )

        def _backward() -> None:
            grad = _grad_data(out)
            if left.requires_grad:
                _add_grad(
                    left,
                    [
                        grad[row] * right[col]
                        for row in range(rows)
                        for col in range(shared)
                    ],
                )
            if right.requires_grad:
                right_grad = [0.0] * shared
                for row in range(rows):
                    for col in range(shared):
                        right_grad[col] += left[row, col] * grad[row]
                _add_grad(right, right_grad)

        out._backward = _backward
        return out

    if len(left.shape) == 2 and len(right.shape) == 2:
        rows, shared = left.shape
        right_rows, cols = right.shape
        if shared != right_rows:
            raise ValueError("matrix-matrix inner dimensions must match")

        data = []
        for row in range(rows):
            for col in range(cols):
                total = 0.0
                for inner in range(shared):
                    total += left[row, inner] * right[inner, col]
                data.append(total)
        out = Tensor(
            data,
            (rows, cols),
            requires_grad=left.requires_grad or right.requires_grad,
            _children=(left, right),
            _op="matmul",
        )

        def _backward() -> None:
            grad = _grad_data(out)
            if left.requires_grad:
                left_grad = [0.0] * len(left.data)
                for row in range(rows):
                    for inner in range(shared):
                        total = 0.0
                        for col in range(cols):
                            total += grad[row * cols + col] * right[inner, col]
                        left_grad[row * shared + inner] += total
                _add_grad(left, left_grad)
            if right.requires_grad:
                right_grad = [0.0] * len(right.data)
                for inner in range(shared):
                    for col in range(cols):
                        total = 0.0
                        for row in range(rows):
                            total += left[row, inner] * grad[row * cols + col]
                        right_grad[inner * cols + col] += total
                _add_grad(right, right_grad)

        out._backward = _backward
        return out

    raise ValueError("matmul does not support vector-matrix multiplication yet")


def transpose(tensor: Tensor) -> Tensor:
    """Transpose a 2D tensor."""

    if len(tensor.shape) != 2:
        raise ValueError("transpose expects a 2D tensor")

    rows, cols = tensor.shape
    data = []
    for col in range(cols):
        for row in range(rows):
            data.append(tensor[row, col])
    out = Tensor(
        data,
        (cols, rows),
        requires_grad=tensor.requires_grad,
        _children=(tensor,),
        _op="transpose",
    )

    def _backward() -> None:
        if not tensor.requires_grad:
            return
        grad = _grad_data(out)
        tensor_grad = [0.0] * len(tensor.data)
        for col in range(cols):
            for row in range(rows):
                tensor_grad[row * cols + col] += grad[col * rows + row]
        _add_grad(tensor, tensor_grad)

    out._backward = _backward
    return out


def tensor_sum(tensor: Tensor, axis: int | None = None) -> Tensor:
    """Sum all values or reduce a 2D tensor along one axis."""

    if axis is None:
        out = Tensor(
            [sum(tensor.data)],
            (1,),
            requires_grad=tensor.requires_grad,
            _children=(tensor,),
            _op="sum",
        )

        def _backward() -> None:
            if tensor.requires_grad:
                _add_grad(tensor, [_grad_data(out)[0]] * len(tensor.data))

        out._backward = _backward
        return out

    if len(tensor.shape) == 1:
        if axis != 0:
            raise ValueError("1D tensors only support axis=0")
        return tensor_sum(tensor)

    rows, cols = tensor.shape
    if axis == 0:
        data = []
        for col in range(cols):
            total = 0.0
            for row in range(rows):
                total += tensor[row, col]
            data.append(total)
        out = Tensor(
            data,
            (cols,),
            requires_grad=tensor.requires_grad,
            _children=(tensor,),
            _op="sum",
        )

        def _backward() -> None:
            if not tensor.requires_grad:
                return
            grad = _grad_data(out)
            _add_grad(
                tensor,
                [
                    grad[col]
                    for row in range(rows)
                    for col in range(cols)
                ],
            )

        out._backward = _backward
        return out

    if axis == 1:
        data = []
        for row in range(rows):
            total = 0.0
            for col in range(cols):
                total += tensor[row, col]
            data.append(total)
        out = Tensor(
            data,
            (rows,),
            requires_grad=tensor.requires_grad,
            _children=(tensor,),
            _op="sum",
        )

        def _backward() -> None:
            if not tensor.requires_grad:
                return
            grad = _grad_data(out)
            _add_grad(
                tensor,
                [
                    grad[row]
                    for row in range(rows)
                    for col in range(cols)
                ],
            )

        out._backward = _backward
        return out

    raise ValueError("2D tensors only support axis=0 or axis=1")


def tensor_mean(tensor: Tensor, axis: int | None = None) -> Tensor:
    """Mean of all values or a 2D tensor along one axis."""

    if axis is None:
        return tensor_sum(tensor) * (1 / len(tensor.data))

    if len(tensor.shape) == 1:
        if axis != 0:
            raise ValueError("1D tensors only support axis=0")
        return tensor_sum(tensor, axis=0) * (1 / tensor.shape[0])

    if axis == 0:
        return tensor_sum(tensor, axis=0) * (1 / tensor.shape[0])

    if axis == 1:
        return tensor_sum(tensor, axis=1) * (1 / tensor.shape[1])

    raise ValueError("2D tensors only support axis=0 or axis=1")


def tensor_exp(tensor: Tensor) -> Tensor:
    """Elementwise exponential."""

    return _map(tensor, math.exp, math.exp, "exp")


def tensor_log(tensor: Tensor) -> Tensor:
    """Elementwise natural logarithm."""

    if any(value <= 0 for value in tensor.data):
        raise ValueError("log is only defined for positive values")
    return _map(tensor, math.log, lambda value: 1 / value, "log")


def tensor_relu(tensor: Tensor) -> Tensor:
    """Elementwise rectified linear unit."""

    return _map(
        tensor,
        lambda value: max(0.0, value),
        lambda value: 1.0 if value > 0 else 0.0,
        "relu",
    )


def tensor_sigmoid(tensor: Tensor) -> Tensor:
    """Elementwise sigmoid."""

    def sigmoid(value: float) -> float:
        return 1 / (1 + math.exp(-value))

    return _map(
        tensor,
        sigmoid,
        lambda value: sigmoid(value) * (1 - sigmoid(value)),
        "sigmoid",
    )


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _normalize_index(index: int, size: int) -> int:
    if index < 0:
        index += size
    if index < 0 or index >= size:
        raise IndexError("tensor index out of range")
    return index


def _require_same_shape(left: Tensor, right: Tensor) -> None:
    if left.shape != right.shape:
        raise ValueError("tensor shapes must match")


def _can_row_broadcast(matrix: Tensor, row: Tensor) -> bool:
    return (
        len(matrix.shape) == 2
        and len(row.shape) == 1
        and matrix.shape[1] == row.shape[0]
    )


def _row_broadcast(matrix: Tensor, row: Tensor, fn, _op: str = "") -> Tensor:
    rows, cols = matrix.shape
    data = []
    for i in range(rows):
        for j in range(cols):
            data.append(fn(matrix[i, j], row[j]))
    return Tensor(
        data,
        matrix.shape,
        requires_grad=matrix.requires_grad or row.requires_grad,
        _children=(matrix, row),
        _op=_op,
    )


def _map(tensor: Tensor, fn, grad_fn, op: str) -> Tensor:
    out = Tensor(
        [fn(value) for value in tensor.data],
        tensor.shape,
        requires_grad=tensor.requires_grad,
        _children=(tensor,),
        _op=op,
    )

    def _backward() -> None:
        if tensor.requires_grad:
            _add_grad(
                tensor,
                [
                    grad_fn(value) * grad
                    for value, grad in zip(tensor.data, _grad_data(out))
                ],
            )

    out._backward = _backward
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
    if tensor.grad is None:
        tensor.grad = [0.0] * len(tensor.data)
    if len(grad) != len(tensor.grad):
        raise ValueError("gradient length does not match tensor data")
    for i, value in enumerate(grad):
        tensor.grad[i] += float(value)


def _sum_row_broadcast_grad(
    matrix_shape: tuple[int, ...],
    grad: Sequence[Number],
) -> list[float]:
    rows, cols = matrix_shape
    row_grad = [0.0] * cols
    for row in range(rows):
        for col in range(cols):
            row_grad[col] += grad[row * cols + col]
    return row_grad
