"""Non-autograd tensor neural-network helpers."""

from __future__ import annotations

from tensor import Tensor, matmul


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
        return _add_row_bias(outputs, bias)

    raise ValueError("linear expects 1D or 2D inputs")


def _add_row_bias(outputs: Tensor, bias: Tensor) -> Tensor:
    rows, cols = outputs.shape
    data = []
    for row in range(rows):
        for col in range(cols):
            data.append(outputs[row, col] + bias[col])
    return Tensor(data, outputs.shape)


def _require_vector(name: str, tensor: Tensor) -> None:
    if len(tensor.shape) != 1:
        raise ValueError(f"{name} must be a 1D tensor")


def _require_matrix(name: str, tensor: Tensor) -> None:
    if len(tensor.shape) != 2:
        raise ValueError(f"{name} must be a 2D tensor")
