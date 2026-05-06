"""Optimization tools for scalar and tensor parameters."""

from __future__ import annotations

import math

from engine import Value
from tensor import Tensor


class SGD:
    """Plain stochastic gradient descent."""

    def __init__(self, parameters: list[Value], lr: float = 0.01) -> None:
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = 0.0

    def step(self) -> None:
        for parameter in self.parameters:
            parameter.data -= self.lr * parameter.grad


class TensorSGD:
    """Plain stochastic gradient descent for tensor parameters."""

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.01,
        *,
        max_grad_norm: float | None = None,
    ) -> None:
        if max_grad_norm is not None and max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        self.parameters = parameters
        self.lr = lr
        self.max_grad_norm = max_grad_norm

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self) -> None:
        grad_scale = _global_grad_scale(self.parameters, self.max_grad_norm)
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            for i, grad in enumerate(parameter.grad):
                parameter.data[i] -= self.lr * grad * grad_scale


class TensorAdam:
    """Adam optimizer for tensor parameters."""

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.001,
        *,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        max_grad_norm: float | None = None,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if beta1 < 0.0 or beta1 >= 1.0:
            raise ValueError("beta1 must be in [0, 1)")
        if beta2 < 0.0 or beta2 >= 1.0:
            raise ValueError("beta2 must be in [0, 1)")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if max_grad_norm is not None and max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")

        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.max_grad_norm = max_grad_norm
        self.step_count = 0
        self._m = [
            [0.0] * parameter.numel
            for parameter in parameters
        ]
        self._v = [
            [0.0] * parameter.numel
            for parameter in parameters
        ]

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self) -> None:
        self.step_count += 1
        grad_scale = _global_grad_scale(self.parameters, self.max_grad_norm)
        beta1_correction = 1.0 - self.beta1 ** self.step_count
        beta2_correction = 1.0 - self.beta2 ** self.step_count

        for parameter_index, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue

            first_moment = self._m[parameter_index]
            second_moment = self._v[parameter_index]
            for i, grad in enumerate(parameter.grad):
                scaled_grad = grad * grad_scale
                first_moment[i] = (
                    self.beta1 * first_moment[i]
                    + (1.0 - self.beta1) * scaled_grad
                )
                second_moment[i] = (
                    self.beta2 * second_moment[i]
                    + (1.0 - self.beta2) * scaled_grad * scaled_grad
                )
                corrected_first = first_moment[i] / beta1_correction
                corrected_second = second_moment[i] / beta2_correction
                parameter.data[i] -= (
                    self.lr
                    * corrected_first
                    / (math.sqrt(corrected_second) + self.eps)
                )


def _global_grad_scale(
    parameters: list[Tensor],
    max_grad_norm: float | None,
) -> float:
    if max_grad_norm is None:
        return 1.0

    grad_norm_sq = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad_norm_sq += sum(grad * grad for grad in parameter.grad)

    grad_norm = math.sqrt(grad_norm_sq)
    if grad_norm == 0.0 or grad_norm <= max_grad_norm:
        return 1.0
    return max_grad_norm / grad_norm
