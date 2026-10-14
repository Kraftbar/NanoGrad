"""Tensor neural-network helpers and small module abstractions."""

from __future__ import annotations

import random
import json
import math
from collections.abc import Sequence
from pathlib import Path

import kernels

from tensor import (
    _set_backward,
    Tensor,
    _add_grad,
    _grad_data,
    _op_result,
    conv2d,
    conv_transpose2d,
    depthwise_conv2d,
    dropout,
    gather_rows,
    matmul,
)


class TensorModule:
    """Base class for tensor modules with trainable parameters.

    Modules start in training mode. ``eval()`` switches a module and every
    module inside it to evaluation mode (which, for now, only turns dropout
    off); ``train()`` switches back.
    """

    training = True

    def train(self, mode: bool = True) -> "TensorModule":
        for module in self.modules():
            module.training = mode
        return self

    def eval(self) -> "TensorModule":
        return self.train(False)

    def modules(self) -> list["TensorModule"]:
        """This module and every module stored in its attributes, recursively."""

        found: list[TensorModule] = [self]
        for value in vars(self).values():
            children = value if isinstance(value, (list, tuple)) else [value]
            for child in children:
                if isinstance(child, TensorModule):
                    found.extend(child.modules())
        return found

    def parameters(self) -> list[Tensor]:
        return []

    def num_parameters(self) -> int:
        return sum(parameter.numel for parameter in self.parameters())

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def state_dict(self) -> dict:
        """Every parameter's values, in ``parameters()`` order, plus the
        running statistics of any batch norm inside. Modules with their own
        named format override this."""

        parameters = self.parameters()
        if not parameters:
            return {}
        return {
            "parameters": [list(parameter.data) for parameter in parameters],
            "buffers": [list(values) for module in self.modules() for values in _buffers(module)],
        }

    def load_state_dict(self, state: dict) -> None:
        parameters = self.parameters()
        if not parameters:
            if state:
                raise ValueError("unexpected state for empty TensorModule")
            return
        saved = state.get("parameters")
        if not isinstance(saved, list) or len(saved) != len(parameters):
            raise ValueError("state does not have one entry per parameter")
        for parameter, values in zip(parameters, saved):
            if len(values) != parameter.numel:
                raise ValueError("state parameter size does not match")
            parameter.data = [float(value) for value in values]
        buffers = [values for module in self.modules() for values in _buffers(module)]
        for target, values in zip(buffers, state.get("buffers", []), strict=True):
            target[:] = [float(value) for value in values]

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


class TensorDepthwiseConv2D(TensorModule):
    """Depthwise convolution: one (k x k) kernel per channel (``tensor.depthwise_conv2d``)."""

    def __init__(self, channels: int, kernel_size: int = 3, *, stride: int = 1, padding: int = 1,
                 seed: int | None = None) -> None:
        self.stride, self.padding = stride, padding
        self.kernel = conv2d_kernel((channels, 1, kernel_size, kernel_size), seed=seed)

    def __call__(self, inputs: Tensor) -> Tensor:
        return depthwise_conv2d(inputs, self.kernel, stride=self.stride, padding=self.padding)

    def parameters(self) -> list[Tensor]:
        return [self.kernel]

    def state_dict(self) -> dict:
        return {"kernel": _tensor_state(self.kernel)}

    def load_state_dict(self, state: dict) -> None:
        self.kernel.data = _state_data(state, "kernel", self.kernel.shape)


def _buffers(module: TensorModule) -> list[list[float]]:
    """State that is not a parameter but must be saved (batch norm's running statistics)."""

    if isinstance(module, TensorBatchNorm2d):
        return [module.running_mean, module.running_var]
    return []


class TensorConvTranspose2D(TensorModule):
    """Transposed convolution layer (``tensor.conv_transpose2d``): kernel (Cin, Cout, k, k)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, *, stride: int = 1,
                 padding: int = 0, use_bias: bool = True, seed: int | None = None) -> None:
        self.stride, self.padding = stride, padding
        self.kernel = conv2d_kernel((in_channels, out_channels, kernel_size, kernel_size), seed=seed)
        self.bias = Tensor.zeros((out_channels, 1, 1), requires_grad=True) if use_bias else None

    def __call__(self, inputs: Tensor) -> Tensor:
        out = conv_transpose2d(inputs, self.kernel, stride=self.stride, padding=self.padding)
        return out if self.bias is None else out + self.bias

    def parameters(self) -> list[Tensor]:
        return [self.kernel] if self.bias is None else [self.kernel, self.bias]


class Sequential(TensorModule):
    """Layers applied one after another: ``Sequential(a, b, c)(x) == c(b(a(x)))``."""

    def __init__(self, *layers: TensorModule) -> None:
        self.layers = list(layers)

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Tensor]:
        return [parameter for layer in self.layers for parameter in layer.parameters()]


class ReLU(TensorModule):
    def __call__(self, x: Tensor) -> Tensor:
        return x.relu()


class LeakyReLU(TensorModule):
    """x for x > 0, slope * x below: relu(x) - slope * relu(-x)."""

    def __init__(self, slope: float = 0.01) -> None:
        self.slope = slope

    def __call__(self, x: Tensor) -> Tensor:
        return x.relu() - (-x).relu() * self.slope


class Tanh(TensorModule):
    def __call__(self, x: Tensor) -> Tensor:
        return x.tanh()


class MaxPool2d(TensorModule):
    """The largest value in each size x size window (stride = size; leftover edges are dropped)."""

    def __init__(self, size: int = 2) -> None:
        self.size = size

    def __call__(self, x: Tensor) -> Tensor:
        return x.max_pool2d((self.size, self.size))


class Flatten(TensorModule):
    """(B, ...) -> (B, everything else)."""

    def __call__(self, x: Tensor) -> Tensor:
        return x.reshape((x.shape[0], x.numel // x.shape[0]))


class TensorConv2D(TensorModule):
    """Convolution layer for early vision experiments."""

    def __init__(
        self,
        kernel_shape: tuple[int, ...],
        *,
        kernel: Tensor | None = None,
        bias: Tensor | None = None,
        seed: int | None = None,
        stride: int = 1,
        padding: int = 0,
        use_bias: bool = True,
    ) -> None:
        if len(kernel_shape) not in (2, 3, 4):
            raise ValueError("TensorConv2D kernel shape must be 2D, 3D, or 4D")
        if any(dim <= 0 for dim in kernel_shape):
            raise ValueError("TensorConv2D kernel dimensions must be positive")

        bias_shape = _conv2d_bias_shape(kernel_shape)
        self.stride = stride
        self.padding = padding
        self.kernel = kernel or conv2d_kernel(kernel_shape, seed=seed)
        # A conv followed by batch norm needs no bias: the norm subtracts it.
        self.bias = (bias or Tensor.zeros(bias_shape, requires_grad=True)) if use_bias else None

        _require_conv2d_kernel("kernel", self.kernel)
        if self.kernel.shape != kernel_shape:
            raise ValueError("kernel shape must match kernel_shape")
        if self.bias is not None and self.bias.shape not in ((1,), bias_shape):
            raise ValueError("bias shape must be (1,) or match conv output channels")

    def __call__(self, inputs: Tensor) -> Tensor:
        out = conv2d(inputs, self.kernel, stride=self.stride, padding=self.padding)
        return out if self.bias is None else out + self.bias

    def parameters(self) -> list[Tensor]:
        return [self.kernel] if self.bias is None else [self.kernel, self.bias]

    def state_dict(self) -> dict:
        state = {"kernel": _tensor_state(self.kernel)}
        if self.bias is not None:
            state["bias"] = _tensor_state(self.bias)
        return state

    def load_state_dict(self, state: dict) -> None:
        self.kernel.data = _state_data(state, "kernel", self.kernel.shape)
        if self.bias is not None:
            self.bias.data = _state_data(state, "bias", self.bias.shape)


class TensorBatchNorm2d(TensorModule):
    """Batch normalization for (batch, channels, rows, cols) images.

    Training mode normalizes each channel by the mean and variance of all its
    values in the batch (every image, every pixel), then scales and shifts
    it with learned per-channel ``weight`` and ``bias``:

        x_hat = (x - mean_c) / sqrt(var_c + eps),    y = x_hat * weight_c + bias_c

    That is LayerNorm with the roles of the axes swapped (normalize each
    channel instead of each row), and its gradient is LayerNorm's too; see
    ``batch_norm``. It also keeps running averages of the statistics for
    evaluation mode, as PyTorch does:

        running = (1 - momentum) * running + momentum * batch statistic

    (the running variance uses the unbiased n / (n - 1) estimate). Evaluation
    mode normalizes with the running statistics instead of the batch's.
    """

    def __init__(self, channels: int, *, eps: float = 1e-5, momentum: float = 0.1) -> None:
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.channels = channels
        self.eps = eps
        self.momentum = momentum
        self.weight = Tensor([1.0] * channels, (channels,), requires_grad=True)
        self.bias = Tensor.zeros((channels,), requires_grad=True)
        self.running_mean = [0.0] * channels
        self.running_var = [1.0] * channels

    def __call__(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.channels:
            raise ValueError("TensorBatchNorm2d expects (batch, channels, rows, cols)")
        batch, channels, rows, cols = inputs.shape
        per_channel = (channels, 1, 1)
        if self.training:
            out, mean, variance = batch_norm(inputs, self.weight, self.bias, eps=self.eps)
            self._update_running_statistics(mean, variance, batch * rows * cols)
            return out
        mean = Tensor(self.running_mean, per_channel)
        inv_std = Tensor([1.0 / math.sqrt(v + self.eps) for v in self.running_var], per_channel)
        x_hat = (inputs - mean) * inv_std
        return x_hat * self.weight.reshape(per_channel) + self.bias.reshape(per_channel)

    def _update_running_statistics(self, mean: Sequence[float], variance: Sequence[float], count: int) -> None:
        momentum = self.momentum
        for channel in range(self.channels):
            unbiased = variance[channel] * count / max(count - 1, 1)
            self.running_mean[channel] = (1 - momentum) * self.running_mean[channel] + momentum * mean[channel]
            self.running_var[channel] = (1 - momentum) * self.running_var[channel] + momentum * unbiased

    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]

    def state_dict(self) -> dict:
        return {
            "weight": _tensor_state(self.weight),
            "bias": _tensor_state(self.bias),
            "running_mean": list(self.running_mean),
            "running_var": list(self.running_var),
        }

    def load_state_dict(self, state: dict) -> None:
        self.weight.data = _state_data(state, "weight", self.weight.shape)
        self.bias.data = _state_data(state, "bias", self.bias.shape)
        for name in ("running_mean", "running_var"):
            values = state.get(name)
            if not isinstance(values, list) or len(values) != self.channels:
                raise ValueError(f"state {name} does not match TensorBatchNorm2d")
            setattr(self, name, [float(value) for value in values])



class TensorDropout(TensorModule):
    """Dropout as a module: active in training mode, identity in eval mode."""

    def __init__(self, p: float = 0.0, *, seed: int | None = None) -> None:
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.p = p
        self.rng = random.Random(seed)

    def __call__(self, inputs: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return inputs
        return dropout(inputs, self.p, self.rng)


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
        embedded = gather_rows(self.weight, flat_indices)
        if len(index_shape) == 1:
            return embedded
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


class TensorLayerNorm(TensorModule):
    """Layer normalization over the last tensor dimension."""

    def __init__(
        self,
        normalized_shape: int,
        *,
        eps: float = 1e-5,
        weight: Tensor | None = None,
        bias: Tensor | None = None,
    ) -> None:
        if normalized_shape <= 0:
            raise ValueError("TensorLayerNorm normalized_shape must be positive")
        if eps <= 0.0:
            raise ValueError("TensorLayerNorm eps must be positive")

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = weight or Tensor(
            [1.0] * normalized_shape,
            (normalized_shape,),
            requires_grad=True,
        )
        self.bias = bias or Tensor.zeros((normalized_shape,), requires_grad=True)

        _require_vector("weight", self.weight)
        _require_vector("bias", self.bias)
        if self.weight.shape != (normalized_shape,):
            raise ValueError("weight shape must be (normalized_shape,)")
        if self.bias.shape != (normalized_shape,):
            raise ValueError("bias shape must be (normalized_shape,)")

    def __call__(self, inputs: Tensor) -> Tensor:
        if inputs.shape[-1] != self.normalized_shape:
            raise ValueError("input last dimension must match normalized_shape")
        return layer_norm(inputs, self.weight, self.bias, eps=self.eps)

    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]

    def state_dict(self) -> dict:
        return {
            "normalized_shape": self.normalized_shape,
            "eps": self.eps,
            "weight": _tensor_state(self.weight),
            "bias": _tensor_state(self.bias),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("normalized_shape") != self.normalized_shape:
            raise ValueError("state normalized_shape does not match TensorLayerNorm")
        if state.get("eps") != self.eps:
            raise ValueError("state eps does not match TensorLayerNorm")
        self.weight.data = _state_data(state, "weight", self.weight.shape)
        self.bias.data = _state_data(state, "bias", self.bias.shape)


class TensorGroupNorm(TensorModule):
    """Group normalization (Wu and He, 2018) of (batch, channels, ...) inputs:
    the channels are split into ``groups``, each image's group is normalized
    to zero mean and unit variance over its channels and positions, then
    every channel gets its own scale and shift. It is a layer norm over the
    rows of a (batch * groups, rest) view, so it reuses that fused kernel."""

    def __init__(self, groups: int, channels: int, *, eps: float = 1e-5) -> None:
        if groups <= 0 or channels % groups:
            raise ValueError("TensorGroupNorm needs a positive group count that divides channels")
        self.groups, self.channels, self.eps = groups, channels, eps
        self.weight = Tensor([1.0] * channels, (channels,), requires_grad=True)
        self.bias = Tensor.zeros((channels,), requires_grad=True)
        self._unit: dict[int, tuple[Tensor, Tensor]] = {}

    def __call__(self, inputs: Tensor) -> Tensor:
        if inputs.shape[1] != self.channels:
            raise ValueError("input channels must match TensorGroupNorm")
        width = inputs.numel // (inputs.shape[0] * self.groups)
        if width not in self._unit:                 # a plain normalization: unit scale, zero shift, no gradient
            self._unit[width] = (Tensor([1.0] * width, (width,)), Tensor.zeros((width,)))
        rows = inputs.reshape((inputs.shape[0] * self.groups, width))
        normed = layer_norm(rows, *self._unit[width], eps=self.eps).reshape(inputs.shape)
        per_channel = (1, self.channels) + (1,) * (len(inputs.shape) - 2)
        return normed * self.weight.reshape(per_channel) + self.bias.reshape(per_channel)

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
    """Create a trainable convolution kernel, initialized like PyTorch's.

    Each weight is uniform in +-1 / sqrt(fan_in), where fan_in is how many
    inputs one output value sums over: channels * kernel rows * kernel cols
    for a (filters, channels, rows, cols) kernel (for 2D/3D kernels, the
    kernel area). That keeps each output's variance about the same as its
    inputs', however many filters the layer has.
    """

    if len(kernel_shape) not in (2, 3, 4):
        raise ValueError("kernel shape must be 2D, 3D, or 4D")
    if any(dim <= 0 for dim in kernel_shape):
        raise ValueError("kernel dimensions must be positive")

    fan_in = _numel(kernel_shape[1:]) if len(kernel_shape) == 4 else _numel(kernel_shape[-2:])
    limit = 1.0 / math.sqrt(fan_in)
    rng = random.Random(seed)
    return Tensor(
        [rng.uniform(-limit, limit) for _ in range(_numel(kernel_shape))],
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
    """Apply a dense linear layer: y = x @ W^T + b.

    Shapes:
    - vector input: inputs=(in,), weight=(out, in), bias=(out,) -> (out,)
    - batch input: inputs=(..., in), weight=(out, in), bias=(out,) -> (..., out)
      Any leading axes (batch, time, ...) are treated as extra rows.
    """

    _require_matrix("weight", weight)
    _require_vector("bias", bias)

    if len(inputs.shape) == 1:
        outputs = matmul(weight, inputs)
        if outputs.shape != bias.shape:
            raise ValueError("bias shape must match linear output shape")
        return outputs + bias

    if inputs.shape[-1] != weight.shape[1]:
        raise ValueError("linear input features must match weight columns")
    if weight.shape[0] != bias.shape[0]:
        raise ValueError("bias shape must match linear output columns")

    # One fused op (PyTorch's addmm): every row of x, times W^T read in
    # place (no transposed copy), added onto a copy of the bias.
    #     dx = dy @ W        dW = dy^T @ x        db = dy summed over rows
    outputs, features = weight.shape
    rows = inputs.numel // features
    data = kernels.gemm(
        inputs.data, weight.data, batches=1, m=rows, k=features, n=outputs,
        a_step=0, b_step=0, c_step=0, trans_b=True, initial=kernels.storage(bias.data) * rows,
    )
    out = _op_result(data, (*inputs.shape[:-1], outputs), (inputs, weight, bias), "linear")

    def _backward() -> None:
        grad = _grad_data(out)
        if inputs.requires_grad:
            _add_grad(inputs, kernels.gemm(grad, weight.data, batches=1, m=rows, k=outputs, n=features,
                                           a_step=0, b_step=0, c_step=0))
        if weight.requires_grad:
            _add_grad(weight, kernels.gemm(grad, inputs.data, batches=1, m=outputs, k=rows, n=features,
                                           a_step=0, b_step=0, c_step=0, trans_a=True))
        if bias.requires_grad:
            _add_grad(bias, kernels.sum_repeated(grad, outer=rows, size=outputs, inner=1))

    _set_backward(out, _backward)
    return out


def layer_norm(
    inputs: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    eps: float = 1e-5,
) -> Tensor:
    """Normalize each last-axis row to zero mean and unit variance.

    Per row x of width n:

        mean    = sum(x) / n
        var     = sum((x - mean)^2) / n
        inv_std = 1 / sqrt(var + eps)
        x_hat   = (x - mean) * inv_std
        y       = x_hat * weight + bias

    Backward, with g = dy * weight for the row:

        dx      = inv_std * (g - mean(g) - x_hat * mean(g * x_hat))
        dweight = sum over rows of dy * x_hat
        dbias   = sum over rows of dy

    This is one fused op instead of ~10 small ones; test_tensor_nn checks it
    against the composed version.
    """

    width = inputs.shape[-1]
    if weight.shape != (width,) or bias.shape != (width,):
        raise ValueError("layer_norm weight and bias must match the last axis")

    data, x_hat, inv_std = kernels.layer_norm_rows(
        inputs.data, weight.data, bias.data, width, eps,
    )
    out = _op_result(data, inputs.shape, (inputs, weight, bias), "layer_norm")

    def _backward() -> None:
        input_grad, weight_grad, bias_grad = kernels.layer_norm_rows_backward(
            _grad_data(out), x_hat, inv_std, weight.data, width, inputs.requires_grad,
        )
        if inputs.requires_grad:
            _add_grad(inputs, input_grad)
        if weight.requires_grad:
            _add_grad(weight, weight_grad)
        if bias.requires_grad:
            _add_grad(bias, bias_grad)

    _set_backward(out, _backward)
    return out


def batch_norm(inputs: Tensor, weight: Tensor, bias: Tensor, *, eps: float = 1e-5):
    """Training-mode batch norm of (batch, channels, rows, cols) images.

    Per channel c, over all n = batch * rows * cols of its values:

        x_hat = (x - mean_c) / sqrt(var_c + eps),    y = x_hat * weight_c + bias_c

    Backward, with g = dy * weight_c (LayerNorm's formula, per channel):

        dx        = inv_std_c * (g - mean(g) - x_hat * mean(g * x_hat))
        dweight_c = sum of dy * x_hat,   dbias_c = sum of dy

    Returns (output, per-channel means, per-channel biased variances).
    """

    batch, channels, rows, cols = inputs.shape
    sizes = dict(batch=batch, channels=channels, plane=rows * cols)
    data, x_hat, inv_std, mean, variance = kernels.batch_norm_channels(
        inputs.data, weight.data, bias.data, eps=eps, **sizes,
    )
    out = _op_result(data, inputs.shape, (inputs, weight, bias), "batch_norm")

    def _backward() -> None:
        input_grad, weight_grad, bias_grad = kernels.batch_norm_channels_backward(
            _grad_data(out), x_hat, inv_std, weight.data, **sizes,
        )
        if inputs.requires_grad:
            _add_grad(inputs, input_grad)
        if weight.requires_grad:
            _add_grad(weight, weight_grad)
        if bias.requires_grad:
            _add_grad(bias, bias_grad)

    _set_backward(out, _backward)
    return out, mean, variance


def rms_norm(inputs: Tensor, weight: Tensor, *, eps: float = 1e-5) -> Tensor:
    """RMSNorm (LLaMA's replacement for LayerNorm) over the last axis.

    Per row x of width n, no mean subtraction and no bias:

        r = 1 / sqrt(sum(x^2) / n + eps),    y = x * r * weight

    Backward, with g = dy * weight for the row:

        dx      = r * (g - x * r^2 * mean(g * x))
        dweight = sum over rows of dy * x * r
    """

    width = inputs.shape[-1]
    if weight.shape != (width,):
        raise ValueError("rms_norm weight must match the last axis")
    data, inv_rms = kernels.rms_norm_rows(inputs.data, weight.data, width, eps)
    out = _op_result(data, inputs.shape, (inputs, weight), "rms_norm")

    def _backward() -> None:
        input_grad, weight_grad = kernels.rms_norm_rows_backward(
            _grad_data(out), inputs.data, inv_rms, weight.data, width,
        )
        if inputs.requires_grad:
            _add_grad(inputs, input_grad)
        if weight.requires_grad:
            _add_grad(weight, weight_grad)

    _set_backward(out, _backward)
    return out


class TensorRMSNorm(TensorModule):
    """RMSNorm over the last dimension, with a learned per-feature weight."""

    def __init__(self, normalized_shape: int, *, eps: float = 1e-5) -> None:
        if normalized_shape <= 0:
            raise ValueError("TensorRMSNorm normalized_shape must be positive")
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = Tensor([1.0] * normalized_shape, (normalized_shape,), requires_grad=True)

    def __call__(self, inputs: Tensor) -> Tensor:
        return rms_norm(inputs, self.weight, eps=self.eps)

    def parameters(self) -> list[Tensor]:
        return [self.weight]

    def state_dict(self) -> dict:
        return {"normalized_shape": self.normalized_shape, "eps": self.eps, "weight": _tensor_state(self.weight)}

    def load_state_dict(self, state: dict) -> None:
        if state.get("normalized_shape") != self.normalized_shape:
            raise ValueError("state normalized_shape does not match TensorRMSNorm")
        self.weight.data = _state_data(state, "weight", self.weight.shape)


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
    if activation == "gelu":
        return tensor.gelu()
    if activation == "sigmoid":
        return tensor.sigmoid()
    if activation == "tanh":
        return tensor.tanh()
    raise ValueError(f"unknown activation: {activation}")


def _tensor_state(tensor: Tensor) -> dict:
    return {
        "shape": list(tensor.shape),
        "data": list(tensor.data),   # a plain list, so it can be saved as JSON
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
