"""Tensor neural-network helpers and small module abstractions."""

from __future__ import annotations

import random
import json
import math
from collections.abc import Sequence
from pathlib import Path

from tensor import Tensor, conv2d_valid, matmul


class TensorModule:
    """Base class for tensor modules with trainable parameters."""

    def parameters(self) -> list[Tensor]:
        return []

    def num_parameters(self) -> int:
        return sum(parameter.numel for parameter in self.parameters())

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        if state:
            raise ValueError("unexpected state for empty TensorModule")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.state_dict(), indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        self.load_state_dict(json.loads(path.read_text(encoding="utf-8")))


class TensorLinear(TensorModule):
    """Dense tensor layer with weight rows and one bias per output."""

    def __init__(
        self,
        inputs: int,
        outputs: int,
        *,
        weight: Tensor | None = None,
        bias: Tensor | None = None,
        seed: int | None = None,
    ) -> None:
        if inputs <= 0 or outputs <= 0:
            raise ValueError("TensorLinear dimensions must be positive")

        self.weight = weight or xavier_uniform(
            inputs,
            outputs,
            seed=seed,
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

    def state_dict(self) -> dict:
        return {
            "weight": _tensor_state(self.weight),
            "bias": _tensor_state(self.bias),
        }

    def load_state_dict(self, state: dict) -> None:
        self.weight.data = _state_data(state, "weight", self.weight.shape)
        self.bias.data = _state_data(state, "bias", self.bias.shape)


class TensorConv2D(TensorModule):
    """Convolution layer for early vision experiments."""

    def __init__(
        self,
        kernel_shape: tuple[int, ...],
        *,
        kernel: Tensor | None = None,
        bias: Tensor | None = None,
        seed: int | None = None,
    ) -> None:
        if len(kernel_shape) not in (2, 3, 4):
            raise ValueError("TensorConv2D kernel shape must be 2D, 3D, or 4D")
        if any(dim <= 0 for dim in kernel_shape):
            raise ValueError("TensorConv2D kernel dimensions must be positive")

        bias_shape = _conv2d_bias_shape(kernel_shape)
        self.kernel = kernel or conv2d_kernel(kernel_shape, seed=seed)
        self.bias = bias or Tensor.zeros(bias_shape, requires_grad=True)

        _require_conv2d_kernel("kernel", self.kernel)
        if self.kernel.shape != kernel_shape:
            raise ValueError("kernel shape must match kernel_shape")
        if self.bias.shape not in ((1,), bias_shape):
            raise ValueError("bias shape must be (1,) or match conv output channels")

    def __call__(self, inputs: Tensor) -> Tensor:
        return conv2d_valid(inputs, self.kernel) + self.bias

    def parameters(self) -> list[Tensor]:
        return [self.kernel, self.bias]

    def state_dict(self) -> dict:
        return {
            "kernel": _tensor_state(self.kernel),
            "bias": _tensor_state(self.bias),
        }

    def load_state_dict(self, state: dict) -> None:
        self.kernel.data = _state_data(state, "kernel", self.kernel.shape)
        self.bias.data = _state_data(state, "bias", self.bias.shape)


class TensorEmbedding(TensorModule):
    """Trainable lookup table for integer token ids."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        *,
        weight: Tensor | None = None,
        seed: int | None = None,
    ) -> None:
        if vocab_size <= 0 or embedding_dim <= 0:
            raise ValueError("TensorEmbedding dimensions must be positive")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight or embedding_uniform(
            vocab_size,
            embedding_dim,
            seed=seed,
        )

        _require_matrix("weight", self.weight)
        if self.weight.shape != (vocab_size, embedding_dim):
            raise ValueError("weight shape must be (vocab_size, embedding_dim)")

    def __call__(self, indices) -> Tensor:
        index_shape, flat_indices = _index_shape_and_flat(indices)
        one_hot = _one_hot_indices(flat_indices, self.vocab_size)
        embedded = matmul(one_hot, self.weight)
        return embedded.reshape((*index_shape, self.embedding_dim))

    def parameters(self) -> list[Tensor]:
        return [self.weight]

    def state_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "weight": _tensor_state(self.weight),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("vocab_size") != self.vocab_size:
            raise ValueError("state vocab_size does not match TensorEmbedding")
        if state.get("embedding_dim") != self.embedding_dim:
            raise ValueError("state embedding_dim does not match TensorEmbedding")
        self.weight.data = _state_data(state, "weight", self.weight.shape)


class TensorMLP(TensorModule):
    """A compact multilayer perceptron for tensor experiments."""

    def __init__(
        self,
        inputs: int,
        layers: list[int],
        *,
        hidden_activation: str = "relu",
        output_activation: str = "linear",
        seed: int | None = None,
    ) -> None:
        if not layers:
            raise ValueError("TensorMLP needs at least one layer")

        sizes = [inputs, *layers]
        self.layers = [
            TensorLinear(
                sizes[i],
                sizes[i + 1],
                seed=None if seed is None else seed + i,
            )
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

    def state_dict(self) -> dict:
        return {
            "layers": [
                layer.state_dict()
                for layer in self.layers
            ],
            "hidden_activation": self.hidden_activation,
            "output_activation": self.output_activation,
        }

    def load_state_dict(self, state: dict) -> None:
        layers = state.get("layers")
        if not isinstance(layers, list) or len(layers) != len(self.layers):
            raise ValueError("state layer count does not match TensorMLP")
        _require_matching_state_value(
            state,
            "hidden_activation",
            self.hidden_activation,
        )
        _require_matching_state_value(
            state,
            "output_activation",
            self.output_activation,
        )

        for layer, layer_state in zip(self.layers, layers):
            layer.load_state_dict(layer_state)


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


def xavier_uniform(inputs: int, outputs: int, *, seed: int | None = None) -> Tensor:
    """Create a trainable weight matrix with Xavier uniform initialization."""

    if inputs <= 0 or outputs <= 0:
        raise ValueError("initializer dimensions must be positive")

    limit = math.sqrt(6.0 / (inputs + outputs))
    rng = random.Random(seed)
    return Tensor(
        [
            rng.uniform(-limit, limit)
            for _ in range(inputs * outputs)
        ],
        (outputs, inputs),
        requires_grad=True,
    )


def conv2d_kernel(kernel_shape: tuple[int, ...], *, seed: int | None = None) -> Tensor:
    """Create a trainable convolution kernel."""

    if len(kernel_shape) not in (2, 3, 4):
        raise ValueError("kernel shape must be 2D, 3D, or 4D")
    if any(dim <= 0 for dim in kernel_shape):
        raise ValueError("kernel dimensions must be positive")

    fan_in = _numel(kernel_shape)
    limit = math.sqrt(6.0 / (fan_in + 1))
    rng = random.Random(seed)
    return Tensor(
        [
            rng.uniform(-limit, limit)
            for _ in range(fan_in)
        ],
        kernel_shape,
        requires_grad=True,
    )


def embedding_uniform(
    vocab_size: int,
    embedding_dim: int,
    *,
    seed: int | None = None,
) -> Tensor:
    """Create a trainable embedding matrix."""

    if vocab_size <= 0 or embedding_dim <= 0:
        raise ValueError("embedding initializer dimensions must be positive")

    limit = math.sqrt(6.0 / (vocab_size + embedding_dim))
    rng = random.Random(seed)
    return Tensor(
        [
            rng.uniform(-limit, limit)
            for _ in range(vocab_size * embedding_dim)
        ],
        (vocab_size, embedding_dim),
        requires_grad=True,
    )


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


def _index_shape_and_flat(indices) -> tuple[tuple[int, ...], list[int]]:
    if isinstance(indices, Tensor):
        return indices.shape, _integer_indices(indices.data)
    if _is_sequence(indices):
        shape = _index_shape(indices)
        return shape, _flatten_indices(indices)
    raise ValueError("embedding indices must be a tensor or sequence")


def _index_shape(indices) -> tuple[int, ...]:
    if not _is_sequence(indices):
        return ()
    if not indices:
        raise ValueError("embedding indices must not be empty")

    first = indices[0]
    if _is_sequence(first):
        child_shape = _index_shape(first)
        for value in indices:
            if not _is_sequence(value) or _index_shape(value) != child_shape:
                raise ValueError("embedding indices must be rectangular")
        return (len(indices), *child_shape)

    if any(_is_sequence(value) for value in indices):
        raise ValueError("embedding indices must be rectangular")
    return (len(indices),)


def _flatten_indices(indices) -> list[int]:
    if _is_sequence(indices):
        values = []
        for value in indices:
            values.extend(_flatten_indices(value))
        return values
    return _integer_indices([indices])


def _integer_indices(values: Sequence[float]) -> list[int]:
    indices = []
    for value in values:
        index = int(value)
        if index != value:
            raise ValueError("embedding indices must be integers")
        indices.append(index)
    return indices


def _one_hot_indices(indices: list[int], vocab_size: int) -> Tensor:
    if not indices:
        raise ValueError("embedding indices must not be empty")
    data = [0.0] * (len(indices) * vocab_size)
    for row, index in enumerate(indices):
        if index < 0 or index >= vocab_size:
            raise ValueError("embedding index out of range")
        data[row * vocab_size + index] = 1.0
    return Tensor(data, (len(indices), vocab_size))


def _is_sequence(value) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _require_vector(name: str, tensor: Tensor) -> None:
    if len(tensor.shape) != 1:
        raise ValueError(f"{name} must be a 1D tensor")


def _require_matrix(name: str, tensor: Tensor) -> None:
    if len(tensor.shape) != 2:
        raise ValueError(f"{name} must be a 2D tensor")


def _require_conv2d_kernel(name: str, tensor: Tensor) -> None:
    if len(tensor.shape) not in (2, 3, 4):
        raise ValueError(f"{name} must be a 2D, 3D, or 4D tensor")


def _conv2d_bias_shape(kernel_shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(kernel_shape) == 4:
        return (kernel_shape[0], 1, 1)
    return (1,)


def _apply_activation(tensor: Tensor, activation: str) -> Tensor:
    if activation == "linear":
        return tensor
    if activation == "relu":
        return tensor.relu()
    if activation == "sigmoid":
        return tensor.sigmoid()
    if activation == "tanh":
        return tensor.tanh()
    raise ValueError(f"unknown activation: {activation}")


def _tensor_state(tensor: Tensor) -> dict:
    return {
        "shape": list(tensor.shape),
        "data": tensor.data[:],
    }


def _state_data(state: dict, name: str, shape: tuple[int, ...]) -> list[float]:
    value = state.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing tensor state: {name}")
    if tuple(value.get("shape", ())) != shape:
        raise ValueError(f"state shape for {name} does not match module shape")
    data = value.get("data")
    if not isinstance(data, list) or len(data) != _numel(shape):
        raise ValueError(f"state data for {name} does not match module shape")
    return [float(item) for item in data]


def _require_matching_state_value(state: dict, name: str, expected: str) -> None:
    value = state.get(name, expected)
    if value != expected:
        raise ValueError(f"state {name} does not match TensorMLP")


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total
