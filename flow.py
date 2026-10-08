"""Normalizing flows: MADE and MAF density estimation on MNIST, NanoGrad vs PyTorch.

A VAE (vae.py) bounds an image's likelihood and a diffusion model
(diffusion.py) learns to denoise. A normalizing flow computes the exact
likelihood. It is an invertible map f from data x to noise u ~ N(0, I).
By the change of variables,

    log p(x) = log N(f(x); 0, I) + log |det df/dx|

so the training loss is just -log p(x) on the data.

MADE (Germain et al., 2015) is an autoregressive network. It is an MLP
whose weights are multiplied by fixed 0/1 masks, so output d only sees
inputs before d in some order. It gives each pixel a Gaussian conditioned
on the earlier ones:

    mu_d, alpha_d = MADE(x_<d)          u_d = (x_d - mu_d) exp(-alpha_d)

The Jacobian du/dx is triangular, with exp(-alpha_d) on the diagonal, so
log |det| = -sum_d alpha_d, from one pass of the network.

MAF (Papamakarios et al., 2017) stacks K MADEs, each in the reverse order
of the last, with batch normalization (an invertible affine map with its
own log-determinant) between them. MAF's MNIST recipe is: 1,024 hidden
units per MADE, Adam at lr 1e-4, batch 100. Pixels are dequantized with
uniform noise and moved to logit space:

    y = logit(lambda + (1 - 2 lambda) (pixel + uniform) / 256),   lambda = 1e-6

The paper reports test log-likelihoods in that space: about -1367 nats
for a single Gaussian, -1381 for MADE, and -1300 for MAF(5). Here the
results are also converted back to pixel space as bits per dimension.

    python3 flow.py --model made --epochs 5
    python3 flow.py --model maf --layers 5 --epochs 20
    .venv/bin/python flow.py --frameworks nanograd torch --epochs 1 --train-limit 2000

Data: data/mnist/ (see data/mnist/README.md).
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from time import perf_counter

from mnist_compare import load_split
from optim import TensorAdam
from tensor import Tensor, no_grad
from tensor_nn import TensorModule, linear, xavier_uniform

LAMBDA = 1e-6
HALF_LOG_TWO_PI = 0.5 * math.log(2 * math.pi)


def to_logit(images: list[bytes], rng: random.Random) -> tuple[list[float], list[float]]:
    """Dequantized pixels in logit space, and log |dy/dpixel| per image (the
    Jacobian that takes a density in logit space back to pixel space)."""

    values, jacobians = [], []
    for image in images:
        total = 0.0
        for pixel in image:
            s = LAMBDA + (1 - 2 * LAMBDA) * (pixel + rng.random()) / 256.0
            values.append(math.log(s / (1 - s)))
            total += math.log((1 - 2 * LAMBDA) / (256.0 * s * (1 - s)))
        jacobians.append(total)
    return values, jacobians


class MaskedLinear(TensorModule):
    """A linear layer whose weight is multiplied by a fixed 0/1 mask."""

    def __init__(self, inputs: int, outputs: int, mask: list[list[int]], rng: random.Random, *,
                 zero: bool = False) -> None:
        self.weight = (Tensor.zeros((outputs, inputs), requires_grad=True) if zero
                       else xavier_uniform(inputs, outputs, seed=rng.randrange(1 << 30)))
        self.bias = Tensor.zeros((outputs,), requires_grad=True)
        self.mask = Tensor([float(m) for row in mask for m in row], (outputs, inputs))

    def __call__(self, x: Tensor) -> Tensor:
        return linear(x, self.weight * self.mask, self.bias)

    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]


class MADE(TensorModule):
    """One autoregressive Gaussian layer: x -> u = (x - mu) exp(-alpha)."""

    def __init__(self, dim: int, hidden: int, order: list[int], rng: random.Random) -> None:
        # order[d] is input d's position (1..dim). A hidden unit gets a degree k in
        # 1..dim-1 and sees inputs at positions <= k; output d sees hidden units
        # of degree < order[d], hence only inputs strictly before it.
        degrees = [rng.randint(1, dim - 1) for _ in range(hidden)]
        into_hidden = [[int(order[d] <= k) for d in range(dim)] for k in degrees]
        into_output = [[int(k < order[d]) for k in degrees] for d in range(dim)]
        self.hidden = MaskedLinear(dim, hidden, into_hidden, rng)
        # mu and alpha start at zero, so each MADE starts as the identity: with
        # logit pixels ranging over about +-14, random ones give exp(-alpha) of e^7.
        self.mu = MaskedLinear(hidden, dim, into_output, rng, zero=True)
        self.alpha = MaskedLinear(hidden, dim, into_output, rng, zero=True)

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """(u, log |det du/dx| per row)."""

        h = self.hidden(x).relu()
        alpha = self.alpha(h)
        return (x - self.mu(h)) * (-alpha).exp(), -alpha.sum(axis=1)

    def parameters(self) -> list[Tensor]:
        return [*self.hidden.parameters(), *self.mu.parameters(), *self.alpha.parameters()]


class BatchNormFlow(TensorModule):
    """y = (x - mean) / sqrt(var + eps) * exp(log_gamma) + beta, per dimension;
    batch statistics in training, running averages in evaluation (MAF's version).
    log |det dy/dx| = sum_d (log_gamma_d - log(var_d + eps) / 2), the same for every row."""

    def __init__(self, dim: int, momentum: float = 0.05, eps: float = 1e-5) -> None:
        self.log_gamma = Tensor.zeros((dim,), requires_grad=True)
        self.beta = Tensor.zeros((dim,), requires_grad=True)
        self.running_mean, self.running_var = [0.0] * dim, [1.0] * dim
        self.momentum, self.eps = momentum, eps

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        dim = x.shape[1]
        if self.training:
            mean = x.mean(axis=0)
            var = ((x - mean) * (x - mean)).mean(axis=0) + self.eps
            m = self.momentum
            self.running_mean = [(1 - m) * r + m * v for r, v in zip(self.running_mean, mean.data)]
            self.running_var = [(1 - m) * r + m * v for r, v in zip(self.running_var, var.data)]
        else:
            mean, var = Tensor(self.running_mean, (dim,)), Tensor(self.running_var, (dim,))
        half_log_var = var.log() * 0.5
        y = (x - mean) * (self.log_gamma - half_log_var).exp() + self.beta
        log_det = (self.log_gamma - half_log_var).sum()
        return y, log_det                         # one number, broadcast over the rows

    def parameters(self) -> list[Tensor]:
        return [self.log_gamma, self.beta]


class MAF(TensorModule):
    """K MADEs in alternating orders, with batch norm between (K = 1 is a plain MADE)."""

    def __init__(self, dim: int = 784, hidden: int = 1024, layers: int = 5, *, batch_norm: bool = True,
                 seed: int = 0) -> None:
        rng = random.Random(seed)
        self.steps: list[TensorModule] = []
        forward = list(range(1, dim + 1))
        for index in range(layers):
            self.steps.append(MADE(dim, hidden, forward if index % 2 == 0 else forward[::-1], rng))
            if batch_norm and index < layers - 1:
                self.steps.append(BatchNormFlow(dim))

    def transform(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """(u = f(x), log |det df/dx| per row)."""

        total = None
        for step in self.steps:
            x, log_det = step(x)
            total = log_det if total is None else total + log_det
        return x, total

    def log_prob(self, x: Tensor) -> Tensor:
        """log p(x) per row: the standard normal density of u, plus the log-determinant."""

        u, log_det = self.transform(x)
        return (u * u).sum(axis=1) * -0.5 - HALF_LOG_TWO_PI * u.shape[1] + log_det

    def parameters(self) -> list[Tensor]:
        return [p for step in self.steps for p in step.parameters()]


class NanoGradRunner:
    def __init__(self, model: MAF, lr: float) -> None:
        self.model = model
        self.optimizer = TensorAdam(model.parameters(), lr=lr)

    def train_step(self, values: list[float], rows: int) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        loss = -self.model.log_prob(Tensor(values, (rows, len(values) // rows))).mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def calibrate(self, values: list[float], rows: int) -> None:
        """Set every batch norm's running statistics to those of these rows."""

        norms = [step for step in self.model.steps if isinstance(step, BatchNormFlow)]
        for norm in norms:
            norm.momentum = 1.0
        self.model.train()
        with no_grad():
            self.model.log_prob(Tensor(values, (rows, len(values) // rows)))
        for norm in norms:
            norm.momentum = 0.05

    def log_prob(self, values: list[float], rows: int) -> list[float]:
        self.model.eval()
        with no_grad():
            return list(self.model.log_prob(Tensor(values, (rows, len(values) // rows))).data)

    def weights(self) -> list[list[float]]:
        return [list(p.data) for p in self.model.parameters()]


class TorchRunner:
    """The same flow in PyTorch, from NanoGrad's weights and masks."""

    def __init__(self, model: MAF, lr: float) -> None:
        import torch

        self.torch = torch

        def t(tensor: Tensor, grad: bool = True):
            return torch.tensor(list(tensor.data), dtype=torch.float64).reshape(tensor.shape).requires_grad_(grad)

        self.steps = []
        for step in model.steps:
            if isinstance(step, MADE):
                self.steps.append(("made", [(t(layer.weight), t(layer.bias), t(layer.mask, False))
                                            for layer in (step.hidden, step.mu, step.alpha)]))
            else:
                self.steps.append(("bn", [t(step.log_gamma), t(step.beta)], step))
        self.parameters = [p for kind, *rest in self.steps for p in
                           ([w for layer in rest[0] for w in layer[:2]] if kind == "made" else rest[0])]
        self.optimizer = torch.optim.Adam(self.parameters, lr=lr)
        dim = model.steps[0].mu.weight.shape[0]
        self.running = [[torch.zeros(dim, dtype=torch.float64), torch.ones(dim, dtype=torch.float64)]
                        for kind, *_ in self.steps if kind == "bn"]
        self.momentum, self.eps = 0.05, 1e-5

    def _log_prob(self, x, training: bool):
        torch = self.torch
        total, bn = 0.0, 0
        for kind, *rest in self.steps:
            if kind == "made":
                (wh, bh, mh), (wm, bm, mm), (wa, ba, ma) = rest[0]
                h = torch.relu(x @ (wh * mh).T + bh)
                alpha = h @ (wa * ma).T + ba
                x = (x - (h @ (wm * mm).T + bm)) * torch.exp(-alpha)
                total = total - alpha.sum(dim=1)
            else:
                log_gamma, beta = rest[0]
                running = self.running[bn]
                bn += 1
                if training:
                    mean = x.mean(dim=0)
                    var = ((x - mean) ** 2).mean(dim=0) + self.eps
                    m = self.momentum
                    running[0] = (1 - m) * running[0] + m * mean.detach()
                    running[1] = (1 - m) * running[1] + m * var.detach()
                else:
                    mean, var = running
                scale = log_gamma - 0.5 * torch.log(var)
                x = (x - mean) * torch.exp(scale) + beta
                total = total + scale.sum()
        return -0.5 * (x * x).sum(dim=1) - HALF_LOG_TWO_PI * x.shape[1] + total

    def train_step(self, values, rows: int) -> float:
        torch = self.torch
        self.optimizer.zero_grad()
        loss = -self._log_prob(torch.tensor(values, dtype=torch.float64).reshape(rows, -1), True).mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def calibrate(self, values, rows: int) -> None:
        torch = self.torch
        self.momentum = 1.0
        with torch.no_grad():
            self._log_prob(torch.tensor(values, dtype=torch.float64).reshape(rows, -1), True)
        self.momentum = 0.05

    def log_prob(self, values, rows: int) -> list[float]:
        torch = self.torch
        with torch.no_grad():
            return self._log_prob(torch.tensor(values, dtype=torch.float64).reshape(rows, -1), False).tolist()

    def weights(self) -> list[list[float]]:
        return [p.detach().reshape(-1).tolist() for p in self.parameters]


def evaluate(runner, images: list[bytes], seed: int, batch: int = 500) -> tuple[float, float]:
    """Mean log-likelihood in logit space (nats per image) and bits per pixel dimension."""

    rng = random.Random(seed)
    logit_total = pixel_total = 0.0
    for first in range(0, len(images), batch):
        chunk = images[first : first + batch]
        values, jacobians = to_logit(chunk, rng)
        for log_p, jacobian in zip(runner.log_prob(values, len(chunk)), jacobians):
            logit_total += log_p
            pixel_total += log_p + jacobian
    count, dim = len(images), len(images[0])
    return logit_total / count, -pixel_total / count / dim / math.log(2)


def train(runner, train_images, test_images, args) -> dict:
    rng = random.Random(args.seed)
    order = list(range(len(train_images)))
    history, steps, training, start = [], 0, 0.0, perf_counter()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(order)
        total, batches, began = 0.0, 0, perf_counter()
        for first in range(0, len(order), args.batch_size):
            chunk = [train_images[i] for i in order[first : first + args.batch_size]]
            values, _ = to_logit(chunk, rng)
            total += runner.train_step(values, len(chunk))
            batches += 1
        steps += batches
        training += perf_counter() - began
        # As in MAF: evaluate with batch norm statistics from training data, not the running averages.
        runner.calibrate(to_logit([train_images[i] for i in order[: args.calibration]], random.Random(args.seed))[0],
                         min(args.calibration, len(order)))
        logit_ll, bits = evaluate(runner, test_images, seed=args.seed + 1)
        history.append((total / batches, logit_ll, bits))
        print(f"  epoch {epoch:2d}: train loss {total / batches:8.1f}, test log-likelihood {logit_ll:8.1f} nats "
              f"(logit space), {bits:.3f} bits/dim, {perf_counter() - start:.0f} s", flush=True)
    return {"history": history, "logit_ll": history[-1][1], "bits": history[-1][2],
            "ms_per_step": 1000 * training / max(1, steps), "seconds": perf_counter() - start}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="data/mnist")
    parser.add_argument("--model", choices=("made", "maf"), default="maf")
    parser.add_argument("--frameworks", nargs="+", choices=("nanograd", "torch"), default=["nanograd"])
    parser.add_argument("--layers", type=int, default=5, help="MADEs in the MAF")
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--calibration", type=int, default=5000,
                        help="training images that set the batch norm statistics before each evaluation")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    train_images, _ = load_split("mnist", Path(args.data), "train", args.train_limit)
    test_images, _ = load_split("mnist", Path(args.data), "test", args.test_limit)
    layers = 1 if args.model == "made" else args.layers
    results, runners = {}, {}
    for framework in args.frameworks:
        model = MAF(len(train_images[0]), args.hidden, layers, seed=args.seed)
        if framework == "torch":
            import torch

            torch.set_num_threads(1)
            runners[framework] = TorchRunner(model, args.lr)
        else:
            runners[framework] = NanoGradRunner(model, args.lr)
        print(f"{framework} {args.model} ({layers} MADE{'s' if layers > 1 else ''} of {args.hidden} hidden units)",
              flush=True)
        results[framework] = result = train(runners[framework], train_images, test_images, args)
        print(f"{framework:8s} {args.model}: test log-likelihood {result['logit_ll']:.1f} nats (logit space), "
              f"{result['bits']:.3f} bits/dim, {result['ms_per_step']:.0f} ms/step", flush=True)
    if len(runners) == 2:
        difference = max(abs(a - b) for x, y in zip(runners["nanograd"].weights(), runners["torch"].weights())
                         for a, b in zip(x, y))
        print(f"largest weight difference after training: {difference:.1e}")
        results["max_weight_difference"] = difference
    return results


if __name__ == "__main__":
    main()
