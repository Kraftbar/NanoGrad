"""Small model definitions assembled from local neural-network blocks."""

from __future__ import annotations

from engine import Value
from nn import Linear, Module


class MLP(Module):
    """A compact multilayer perceptron for scalar-autograd experiments."""

    def __init__(self, inputs: int, layers: list[int]) -> None:
        sizes = [inputs, *layers]
        self.layers = [
            Linear(
                sizes[i],
                sizes[i + 1],
                activation="linear" if i == len(layers) - 1 else "tanh",
            )
            for i in range(len(layers))
        ]

    def __call__(self, x: list[float | Value]) -> Value | list[Value]:
        out: Value | list[Value] = x
        for layer in self.layers:
            inputs = out if isinstance(out, list) else [out]
            out = layer(inputs)
        return out

    def parameters(self) -> list[Value]:
        return [
            parameter
            for layer in self.layers
            for parameter in layer.parameters()
        ]
