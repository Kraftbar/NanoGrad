"""Optimization tools for scalar and tensor parameters."""

from __future__ import annotations

import math

import kernels
import kernels_cuda
from engine import Value
from tensor import RowGrad, Tensor


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
    """Stochastic gradient descent for tensor parameters, with optional
    momentum (PyTorch's ``torch.optim.SGD`` rules, so runs can be compared):

        g   = grad (+ weight_decay * w)
        buf = momentum * buf + g              (buf starts as the first g)
        d   = g + momentum * buf  if nesterov  else  buf
        w   = w - lr * d

    With momentum 0 this is plain SGD: w = w - lr * g.
    """

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.01,
        *,
        momentum: float = 0.0,
        nesterov: bool = False,
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
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if nesterov and momentum == 0.0:
            raise ValueError("nesterov needs momentum")
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.weight_decay_min_ndim = weight_decay_min_ndim
        self.max_grad_norm = max_grad_norm
        self._buffers: list = [None] * len(parameters)

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self) -> None:
        grad_scale = _global_grad_scale(self.parameters, self.max_grad_norm)
        for index, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue
            decays = _uses_weight_decay(parameter, self.weight_decay, self.weight_decay_min_ndim)
            if isinstance(parameter.grad, RowGrad):     # sparse rows: update just those, in place
                if self.momentum or decays:
                    raise ValueError("row-sparse gradients support plain SGD only")
                grad = parameter.grad
                kernels.scatter_add_rows(kernels.scale(grad.values, -self.lr * grad_scale), kernels.index_table(grad.ids),
                                         width=grad.width, rows=grad.rows, into=parameter.data)
                continue
            if not self.momentum and not decays:        # plain SGD: w - lr * scale * grad, in one pass
                parameter.data = kernels.add_scaled(parameter.data, parameter.grad, -self.lr * grad_scale)
                continue
            g = parameter.grad if grad_scale == 1.0 else kernels.scale(parameter.grad, grad_scale)
            if decays:
                g = kernels.add_scaled(g, parameter.data, self.weight_decay)
            direction = g
            if self.momentum:
                buffer = self._buffers[index]
                buffer = g if buffer is None else kernels.add_scaled(g, buffer, self.momentum)
                self._buffers[index] = buffer
                direction = kernels.add_scaled(g, buffer, self.momentum) if self.nesterov else buffer
            parameter.data = kernels.add_scaled(parameter.data, direction, -self.lr)


class TensorAdam:
    """Adam optimizer for tensor parameters.

    ``weight_decay`` is decoupled (AdamW: the weights shrink directly);
    ``l2`` is the older form that ``torch.optim.Adam(weight_decay=...)``
    uses: ``l2 * w`` is added to the gradient before the moments.
    """

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
        l2: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if l2 < 0.0:
            raise ValueError("l2 must be non-negative")
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
        self.l2 = l2
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
        if kernels.capturing():
            self._graphed_step()
            return
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
            grad, scale = parameter.grad, grad_scale
            if isinstance(grad, RowGrad):                 # Adam moves every row (its moments decay), so go dense
                grad = grad.to_dense()
            if self.l2:                                   # g <- g + l2 * w  (torch.optim.Adam's weight_decay)
                grad, scale = kernels.add_scaled(kernels.scale(grad, grad_scale), parameter.data, self.l2), 1.0
            parameter.data, self._m[parameter_index], self._v[parameter_index] = kernels.adam_update(
                parameter.data,
                grad,
                first_moment,
                second_moment,
                grad_scale=scale,
                beta1=self.beta1,
                beta2=self.beta2,
                decay=decay,
                step_size=self.lr / beta1_correction,
                sqrt_correction=math.sqrt(beta2_correction),
                eps=self.eps,
            )

    def _graphed_step(self) -> None:
        """``step`` as a CUDA graph records it (``kernels_cuda.capture``).

        A graph replays its kernels with the arguments they were recorded
        with, so the numbers that change from step to step go through GPU
        memory instead: the step size and bias correction (they depend on
        the step count, and on the learning rate, which a schedule may
        change) are worked out on the host before every replay, by the same
        formulas as in ``step``; the clipping factor is computed on the GPU;
        and the weights are updated in place, so that they keep their
        addresses. The arithmetic is ``step``'s, so the results are the same
        bit for bit.
        """

        def scalars() -> list[float]:                     # called before every replay
            self.step_count += 1
            beta1_correction = 1.0 - self.beta1 ** self.step_count
            beta2_correction = 1.0 - self.beta2 ** self.step_count
            return [self.lr / beta1_correction, math.sqrt(beta2_correction), 1.0 - self.lr * self.weight_decay]

        if self.l2 and self.max_grad_norm is not None:
            raise NotImplementedError("a CUDA graph of Adam with both l2 and max_grad_norm")
        settings = kernels_cuda.per_replay(scalars, 3)
        clip = None
        if self.max_grad_norm is not None:
            grads = [parameter.grad for parameter in self.parameters if parameter.grad is not None]
            clip = kernels_cuda.clip_factor(
                [grad.coalesced().values if isinstance(grad, RowGrad) else grad for grad in grads],
                self.max_grad_norm)
        for parameter_index, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue
            grad = parameter.grad
            if isinstance(grad, RowGrad):
                grad = grad.to_dense()
            if self.l2:                                   # (no clipping, so step's scale(grad, 1.0) is grad)
                grad = kernels.add_scaled(grad, parameter.data, self.l2)
            self._m[parameter_index], self._v[parameter_index] = kernels_cuda.adam_update_graphed(
                parameter.data,
                grad,
                self._m[parameter_index],
                self._v[parameter_index],
                settings,
                clip,
                _uses_weight_decay(parameter, self.weight_decay, self.weight_decay_min_ndim),
                self.beta1,
                self.beta2,
                self.eps,
            )


def _global_grad_scale(
    parameters: list[Tensor],
    max_grad_norm: float | None,
) -> float:
    if max_grad_norm is None:
        return 1.0

    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    grad_norm_sq = kernels.total_sum_of_squares(
        [grad.coalesced().values if isinstance(grad, RowGrad) else grad for grad in grads])

    # PyTorch's clip_grad_norm_ rule (with its 1e-6), so runs can be compared.
    return min(1.0, max_grad_norm / (math.sqrt(grad_norm_sq) + 1e-6))


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


class TensorLBFGS:
    """L-BFGS, a quasi-Newton method, as ``torch.optim.LBFGS`` (no line search).

    Gradient descent steps along -grad. Newton's method steps along
    -H^-1 grad, using the curvature (Hessian H) to take long steps where the
    loss is flat and short ones where it is steep. L-BFGS estimates H^-1 grad
    from the last ``history_size`` steps alone: each pair (s = step taken,
    y = change in gradient) says how the gradient bends along s. The
    "two-loop recursion" turns those pairs into a direction without ever
    building a matrix.

    It works on the whole loss, not on noisy batches: each ``step(closure)``
    runs up to ``max_iter`` iterations, calling ``closure()`` (which must
    zero the gradients, compute the loss, call backward and return it)
    after every move. The arithmetic follows PyTorch's implementation, so
    the two take the same steps.
    """

    def __init__(self, parameters: list[Tensor], lr: float = 1.0, *, max_iter: int = 20, max_eval: int | None = None,
                 tolerance_grad: float = 1e-7, tolerance_change: float = 1e-9, history_size: int = 100) -> None:
        self.parameters = parameters
        self.lr, self.max_iter, self.history_size = lr, max_iter, history_size
        self.max_eval = max_iter * 5 // 4 if max_eval is None else max_eval
        self.tolerance_grad, self.tolerance_change = tolerance_grad, tolerance_change
        self.state: dict = {"n_iter": 0}

    def _flat_grad(self) -> list[float]:
        return [g for parameter in self.parameters
                for g in (parameter.grad if parameter.grad is not None else [0.0] * parameter.numel)]

    def _move(self, step: float, direction: list[float]) -> None:
        offset = 0
        for parameter in self.parameters:
            count = parameter.numel
            parameter.data = kernels.add_scaled(parameter.data, direction[offset : offset + count], step)
            offset += count

    def step(self, closure) -> float:
        state = self.state
        loss = float(closure())
        orig_loss, current_evals = loss, 1
        flat_grad = self._flat_grad()
        if max(abs(g) for g in flat_grad) <= self.tolerance_grad:
            return orig_loss

        d, t = state.get("d"), state.get("t")
        old_dirs, old_steps, ro = state.get("old_dirs"), state.get("old_steps"), state.get("ro")
        h_diag, prev_flat_grad = state.get("h_diag"), state.get("prev_flat_grad")
        n_iter = 0
        while n_iter < self.max_iter:
            n_iter += 1
            state["n_iter"] += 1
            if state["n_iter"] == 1:
                d = [-g for g in flat_grad]
                old_dirs, old_steps, ro, h_diag = [], [], [], 1.0
            else:
                y = kernels.add_scaled(flat_grad, prev_flat_grad, -1.0)       # change in gradient
                s = kernels.scale(d, t)                                         # the step just taken
                ys = _dot(y, s)
                if ys > 1e-10:                                                  # keep only curvature that makes sense
                    if len(old_dirs) == self.history_size:
                        old_dirs.pop(0), old_steps.pop(0), ro.pop(0)
                    old_dirs.append(y)
                    old_steps.append(s)
                    ro.append(1.0 / ys)
                    h_diag = ys / _dot(y, y)                                    # scale of the initial H^-1
                # Two-loop recursion: q = -grad, newest pair first, then back.
                q, alphas = [-g for g in flat_grad], [0.0] * len(old_dirs)
                for i in range(len(old_dirs) - 1, -1, -1):
                    alphas[i] = _dot(old_steps[i], q) * ro[i]
                    q = kernels.add_scaled(q, old_dirs[i], -alphas[i])
                d = kernels.scale(q, h_diag)
                for i in range(len(old_dirs)):
                    beta = _dot(old_dirs[i], d) * ro[i]
                    d = kernels.add_scaled(d, old_steps[i], alphas[i] - beta)
            prev_flat_grad, prev_loss = list(flat_grad), loss
            # The first step is cautious (the direction is just -grad).
            t = min(1.0, 1.0 / sum(abs(g) for g in flat_grad)) * self.lr if state["n_iter"] == 1 else self.lr
            if _dot(flat_grad, d) > -self.tolerance_change:                    # not a descent direction
                break
            self._move(t, d)
            evals = 0
            if n_iter != self.max_iter:
                loss = float(closure())
                flat_grad = self._flat_grad()
                evals = 1
            current_evals += evals
            if n_iter == self.max_iter or current_evals >= self.max_eval:
                break
            if max(abs(g) for g in flat_grad) <= self.tolerance_grad:
                break
            if max(abs(value) for value in d) * abs(t) <= self.tolerance_change:
                break
            if abs(loss - prev_loss) < self.tolerance_change:
                break
        state.update(d=d, t=t, old_dirs=old_dirs, old_steps=old_steps, ro=ro, h_diag=h_diag,
                     prev_flat_grad=prev_flat_grad)
        return orig_loss


def _dot(a, b) -> float:
    return math.fsum(kernels.multiply(a, b))
