"""Optimization tools for scalar and tensor parameters."""

from __future__ import annotations

import math

import kernels
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
        weight_decay: float = 0.0,
        weight_decay_min_ndim: int | None = None,
        max_grad_norm: float | None = None,
    ) -> None:
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if weight_decay_min_ndim is not None and weight_decay_min_ndim <= 0:
            raise ValueError("weight_decay_min_ndim must be positive")
        if max_grad_norm is not None and max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        self.parameters = parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.weight_decay_min_ndim = weight_decay_min_ndim
        self.max_grad_norm = max_grad_norm

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self) -> None:
        grad_scale = _global_grad_scale(self.parameters, self.max_grad_norm)
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            use_weight_decay = _uses_weight_decay(
                parameter,
                self.weight_decay,
                self.weight_decay_min_ndim,
            )
            # w <- w * (1 - lr * weight_decay) - lr * clipped_grad
            decay = 1.0 - self.lr * self.weight_decay if use_weight_decay else 1.0
            step = self.lr * grad_scale
            parameter.data = [
                value * decay - step * grad
                for value, grad in zip(parameter.data, parameter.grad)
            ]


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
        weight_decay: float = 0.0,
        weight_decay_min_ndim: int | None = None,
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
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if weight_decay_min_ndim is not None and weight_decay_min_ndim <= 0:
            raise ValueError("weight_decay_min_ndim must be positive")
        if max_grad_norm is not None and max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")

        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.weight_decay_min_ndim = weight_decay_min_ndim
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
            use_weight_decay = _uses_weight_decay(
                parameter,
                self.weight_decay,
                self.weight_decay_min_ndim,
            )
            # m <- beta1 * m + (1 - beta1) * g          (running mean)
            # v <- beta2 * v + (1 - beta2) * g^2        (running mean of g^2)
            # w <- w * (1 - lr * weight_decay) - lr * m_hat / (sqrt(v_hat) + eps)
            # with bias corrections m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t)
            # (one fused loop per parameter, see kernels.adam_update)
            decay = 1.0 - self.lr * self.weight_decay if use_weight_decay else 1.0
            parameter.data, self._m[parameter_index], self._v[parameter_index] = kernels.adam_update(
                parameter.data,
                parameter.grad,
                first_moment,
                second_moment,
                grad_scale=grad_scale,
                beta1=self.beta1,
                beta2=self.beta2,
                decay=decay,
                step_size=self.lr / beta1_correction,
                sqrt_correction=math.sqrt(beta2_correction),
                eps=self.eps,
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
        grad_norm_sq += math.fsum(kernels.multiply(parameter.grad, parameter.grad))

    grad_norm = math.sqrt(grad_norm_sq)
    if grad_norm == 0.0 or grad_norm <= max_grad_norm:
        return 1.0
    return max_grad_norm / grad_norm


def _uses_weight_decay(
    parameter: Tensor,
    weight_decay: float,
    weight_decay_min_ndim: int | None,
) -> bool:
    if not weight_decay:
        return False
    if weight_decay_min_ndim is None:
        return True
    return parameter.ndim >= weight_decay_min_ndim
