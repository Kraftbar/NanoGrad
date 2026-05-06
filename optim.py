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
        grad_scale = self._grad_scale()
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            for i, grad in enumerate(parameter.grad):
                parameter.data[i] -= self.lr * grad * grad_scale

    def _grad_scale(self) -> float:
        if self.max_grad_norm is None:
            return 1.0

        grad_norm_sq = 0.0
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            grad_norm_sq += sum(grad * grad for grad in parameter.grad)

        grad_norm = math.sqrt(grad_norm_sq)
        if grad_norm == 0.0 or grad_norm <= self.max_grad_norm:
            return 1.0
        return self.max_grad_norm / grad_norm
