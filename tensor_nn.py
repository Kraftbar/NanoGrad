"""Tensor neural-network helpers and small module abstractions."""

from __future__ import annotations

import random

from tensor import Tensor, matmul


class TensorModule:
    """Base class for tensor modules with trainable parameters."""

    def parameters(self) -> list[Tensor]:
        return []

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()


class TensorLinear(TensorModule):
    """Dense tensor layer with weight rows and one bias per output."""

    def __init__(
        self,
        inputs: int,
        outputs: int,
        *,
        weight: Tensor | None = None,
        bias: Tensor | None = None,
    ) -> None:
        if inputs <= 0 or outputs <= 0:
            raise ValueError("TensorLinear dimensions must be positive")

        self.weight = weight or Tensor(
            [
                random.uniform(-1.0, 1.0)
                for _ in range(inputs * outputs)
            ],
            (outputs, inputs),
            requires_grad=True,
        )
        self.bias = bias or Tensor.zeros((outputs,), requires_grad=True)

        _require_matrix("weight", self.weight)
        _require_vector("bias", self.bias)
        if self.weight.shape != (outputs, inputs):
            raise ValueError("weight shape must be (outputs, inputs)")
        if self.bias.shape != (outputs,):
            raise ValueError("bias shape must be (outputs,)")

    def __call__(self, inputs: Tensor) -> Tensor:
        return linear(inputs, self.weight, self.bias)

    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]


class TensorMLP(TensorModule):
    """A compact multilayer perceptron for tensor experiments."""

    def __init__(
        self,
        inputs: int,
        layers: list[int],
        *,
        hidden_activation: str = "relu",
        output_activation: str = "linear",
    ) -> None:
        if not layers:
            raise ValueError("TensorMLP needs at least one layer")

        sizes = [inputs, *layers]
        self.layers = [
            TensorLinear(sizes[i], sizes[i + 1])
            for i in range(len(layers))
        ]
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def __call__(self, inputs: Tensor) -> Tensor:
        out = inputs
        for i, layer in enumerate(self.layers):
            out = layer(out)
            activation = (
                self.output_activation
                if i == len(self.layers) - 1
                else self.hidden_activation
            )
            out = _apply_activation(out, activation)
        return out

    def parameters(self) -> list[Tensor]:
        return [
            parameter
            for layer in self.layers
            for parameter in layer.parameters()
        ]


def binary_mlp(
    inputs: Tensor,
    hidden_weight: Tensor,
    hidden_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
) -> Tensor:
    """Run a one-hidden-layer MLP for binary classification.

    Shapes:
    - vector input: inputs=(in,) -> (1,)
    - batch input: inputs=(batch, in) -> (batch, 1)
    """

    hidden = linear(inputs, hidden_weight, hidden_bias).relu()
    logits = linear(hidden, output_weight, output_bias)
    return logits.sigmoid()


def linear(inputs: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """Apply a dense linear layer.

    Shapes:
    - vector input: inputs=(in,), weight=(out, in), bias=(out,) -> (out,)
    - batch input: inputs=(batch, in), weight=(out, in), bias=(out,) -> (batch, out)
    """

    _require_matrix("weight", weight)
    _require_vector("bias", bias)

    if len(inputs.shape) == 1:
        outputs = matmul(weight, inputs)
        if outputs.shape != bias.shape:
            raise ValueError("bias shape must match linear output shape")
        return outputs + bias

    if len(inputs.shape) == 2:
        outputs = matmul(inputs, weight.T)
        if outputs.shape[1] != bias.shape[0]:
            raise ValueError("bias shape must match linear output columns")
        return outputs + bias

    raise ValueError("linear expects 1D or 2D inputs")


def _require_vector(name: str, tensor: Tensor) -> None:
    if len(tensor.shape) != 1:
        raise ValueError(f"{name} must be a 1D tensor")


def _require_matrix(name: str, tensor: Tensor) -> None:
    if len(tensor.shape) != 2:
        raise ValueError(f"{name} must be a 2D tensor")


def _apply_activation(tensor: Tensor, activation: str) -> Tensor:
    if activation == "linear":
        return tensor
    if activation == "relu":
        return tensor.relu()
    if activation == "sigmoid":
        return tensor.sigmoid()
    raise ValueError(f"unknown activation: {activation}")
