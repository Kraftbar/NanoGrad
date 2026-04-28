"""Small tensor container and shape-checked math helpers.

This module is intentionally non-autograd for now. It gives NanoGrad a clear
place to work out shape, indexing, and matrix math before connecting tensor
operations to the scalar autograd engine.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


Number = int | float


class Tensor:
    """A dense 1D or 2D tensor backed by flat row-major data."""

    def __init__(self, data: Sequence[Number], shape: tuple[int, ...]) -> None:
        if len(shape) not in (1, 2):
            raise ValueError("Tensor currently supports only 1D or 2D shapes")
        if any(dim <= 0 for dim in shape):
            raise ValueError("shape dimensions must be positive")
        if len(data) != _numel(shape):
            raise ValueError("data length does not match shape")

        self.data = [float(value) for value in data]
        self.shape = shape

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
    def zeros(cls, shape: tuple[int, ...]) -> "Tensor":
        return cls([0.0] * _numel(shape), shape)

    @classmethod
    def from_list(cls, values: Sequence[Number] | Sequence[Sequence[Number]]) -> "Tensor":
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
            return cls(data, (len(rows), width))

        return cls(values, (len(values),))

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


def add(left: Tensor, right: Number | Tensor) -> Tensor:
    """Elementwise addition with scalar broadcasting."""

    if isinstance(right, (int, float)):
        return Tensor([value + right for value in left.data], left.shape)

    _require_same_shape(left, right)
    return Tensor(
        [
            a + b
            for a, b in zip(left.data, right.data)
        ],
        left.shape,
    )


def multiply(left: Tensor, right: Number | Tensor) -> Tensor:
    """Elementwise multiplication with scalar broadcasting."""

    if isinstance(right, (int, float)):
        return Tensor([value * right for value in left.data], left.shape)

    _require_same_shape(left, right)
    return Tensor(
        [
            a * b
            for a, b in zip(left.data, right.data)
        ],
        left.shape,
    )


def matmul(left: Tensor, right: Tensor) -> Tensor:
    """Matrix/vector multiplication for 1D and 2D tensors."""

    if len(left.shape) == 1 and len(right.shape) == 1:
        if left.shape[0] != right.shape[0]:
            raise ValueError("vector dot product shapes must match")
        total = sum(
            a * b
            for a, b in zip(left.data, right.data)
        )
        return Tensor([total], (1,))

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
        return Tensor(data, (rows,))

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
        return Tensor(data, (rows, cols))

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
    return Tensor(data, (cols, rows))


def tensor_sum(tensor: Tensor, axis: int | None = None) -> Tensor:
    """Sum all values or reduce a 2D tensor along one axis."""

    if axis is None:
        return Tensor([sum(tensor.data)], (1,))

    if len(tensor.shape) == 1:
        if axis != 0:
            raise ValueError("1D tensors only support axis=0")
        return Tensor([sum(tensor.data)], (1,))

    rows, cols = tensor.shape
    if axis == 0:
        data = []
        for col in range(cols):
            total = 0.0
            for row in range(rows):
                total += tensor[row, col]
            data.append(total)
        return Tensor(data, (cols,))

    if axis == 1:
        data = []
        for row in range(rows):
            total = 0.0
            for col in range(cols):
                total += tensor[row, col]
            data.append(total)
        return Tensor(data, (rows,))

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

    return _map(tensor, math.exp)


def tensor_log(tensor: Tensor) -> Tensor:
    """Elementwise natural logarithm."""

    if any(value <= 0 for value in tensor.data):
        raise ValueError("log is only defined for positive values")
    return _map(tensor, math.log)


def tensor_relu(tensor: Tensor) -> Tensor:
    """Elementwise rectified linear unit."""

    return _map(tensor, lambda value: max(0.0, value))


def tensor_sigmoid(tensor: Tensor) -> Tensor:
    """Elementwise sigmoid."""

    return _map(tensor, lambda value: 1 / (1 + math.exp(-value)))


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


def _map(tensor: Tensor, fn) -> Tensor:
    return Tensor([fn(value) for value in tensor.data], tensor.shape)
