"""Scalar automatic differentiation core.

The first engine target is intentionally small: scalar values with reverse-mode
autograd. Tensor operations can grow out of this once the basic graph mechanics
are easy to inspect and test.
"""

from __future__ import annotations

import math
from typing import Callable


class Value:
    """A scalar value that records the operation that produced it."""

    def __init__(
        self,
        data: float,
        _children: tuple["Value", ...] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        self.data = float(data)
        self.grad = 0.0
        self.label = label

        self._prev = set(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data:g}, grad={self.grad:g})"

    def __add__(self, other: float | "Value") -> "Value":
        other = _as_value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: float | "Value") -> "Value":
        return self + other

    def __neg__(self) -> "Value":
        return self * -1

    def __sub__(self, other: float | "Value") -> "Value":
        return self + -_as_value(other)

    def __rsub__(self, other: float | "Value") -> "Value":
        return _as_value(other) + -self

    def __mul__(self, other: float | "Value") -> "Value":
        other = _as_value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: float | "Value") -> "Value":
        return self * other

    def __pow__(self, power: float) -> "Value":
        out = Value(self.data**power, (self,), f"**{power:g}")

        def _backward() -> None:
            self.grad += power * self.data ** (power - 1) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other: float | "Value") -> "Value":
        return self * _as_value(other) ** -1

    def __rtruediv__(self, other: float | "Value") -> "Value":
        return _as_value(other) * self**-1

    def tanh(self) -> "Value":
        value = math.tanh(self.data)
        out = Value(value, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1 - value * value) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        value = math.exp(self.data)
        out = Value(value, (self,), "exp")

        def _backward() -> None:
            self.grad += value * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Value":
        if self.data <= 0:
            raise ValueError("log is only defined for positive values")

        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "Value":
        value = 1 / (1 + math.exp(-self.data))
        out = Value(value, (self,), "sigmoid")

        def _backward() -> None:
            self.grad += value * (1 - value) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward() -> None:
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        topo: list[Value] = []
        visited: set[Value] = set()

        def build(node: Value) -> None:
            if node in visited:
                return
            visited.add(node)
            for child in node._prev:
                build(child)
            topo.append(node)

        build(self)
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()


def _as_value(value: float | Value) -> Value:
    return value if isinstance(value, Value) else Value(value)
