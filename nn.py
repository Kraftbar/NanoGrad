"""Small neural-network building blocks built on the scalar engine."""

from __future__ import annotations

import random
from typing import Iterable

from engine import Value


class Module:
    """Base class for objects with trainable scalar parameters."""

    def parameters(self) -> list[Value]:
        return []

    def num_parameters(self) -> int:
        return len(self.parameters())

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.grad = 0.0


class Neuron(Module):
    """A single fully connected neuron."""

    def __init__(self, inputs: int, *, activation: str = "tanh") -> None:
        self.w = [Value(random.uniform(-1.0, 1.0)) for _ in range(inputs)]
        self.b = Value(0.0)
        self.activation = activation

    def __call__(self, x: Iterable[float | Value]) -> Value:
        z = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        if self.activation == "tanh":
            return z.tanh()
        if self.activation == "relu":
            return z.relu()
        if self.activation == "linear":
            return z
        raise ValueError(f"unknown activation: {self.activation}")

    def parameters(self) -> list[Value]:
        return [*self.w, self.b]


class Linear(Module):
    """A dense layer represented as rows of neurons."""

    def __init__(
        self,
        inputs: int,
        outputs: int,
        *,
        activation: str = "tanh",
    ) -> None:
        self.neurons = [
            Neuron(inputs, activation=activation)
            for _ in range(outputs)
        ]

    def __call__(self, x: Iterable[float | Value]) -> Value | list[Value]:
        x = list(x)
        out = [neuron(x) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self) -> list[Value]:
        return [
            parameter
            for neuron in self.neurons
            for parameter in neuron.parameters()
        ]
