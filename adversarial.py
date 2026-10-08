"""Adversarial robustness on MNIST: FGSM and PGD attacks, and fast adversarial training.

A classifier that is 99% accurate on MNIST can be fooled by changes too
small to see. Move every pixel by at most eps = 0.3 (of the 0-1 range) in
the direction that raises the loss, and accuracy drops to almost nothing.
The two standard attacks use only the gradient of the loss with respect
to the input image, which autograd computes like any other gradient:

    FGSM (Goodfellow et al., 2015):  x' = x + eps * sign(dL/dx)
    PGD  (Madry et al., 2018):       repeat x' = clip(x' + alpha * sign(dL/dx')) many times,
                                     from a random start in the eps-box, keeping
                                     |x' - x| <= eps and x' in [0, 1]

The defence is adversarial training: train on attacked images instead of
clean ones. Madry et al. attack each batch with 40-step PGD, which makes
training 40 times slower. Wong, Rice and Kolter (2020, "Fast is better
than free") showed that one FGSM step from a random start, with step size
alpha = 1.25 eps, works about as well. This is their MNIST recipe: a
small CNN (two 4x4 stride-2 convolutions, then 100 hidden units), Adam
with the learning rate rising to 5e-3 and falling back over 10 epochs,
batch 100. It is evaluated with PGD of 50 steps of 0.01, from 10 random
starts. The paper reports about 98% clean and 89% accuracy under that
attack at eps = 0.3.

    python3 adversarial.py --train fgsm             # fast adversarial training, then attack it
    python3 adversarial.py --train none --epochs 2  # a normal model, for contrast
    .venv/bin/python adversarial.py --frameworks nanograd torch --epochs 1 --train-limit 2000 --test-limit 200

Data: data/mnist/.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from time import perf_counter

import kernels
from losses import softmax_cross_entropy
from mnist_compare import load_split
from optim import TensorAdam
from tensor import Tensor, no_grad
from tensor_nn import Flatten, ReLU, Sequential, TensorConv2D, TensorLinear

PIXELS = 28 * 28


def small_cnn(seed: int = 0) -> Sequential:
    """Wong et al.'s MNIST network."""

    return Sequential(
        TensorConv2D((16, 1, 4, 4), stride=2, padding=1, seed=seed), ReLU(),           # 28 -> 14
        TensorConv2D((32, 16, 4, 4), stride=2, padding=1, seed=seed + 1), ReLU(),      # 14 -> 7
        Flatten(), TensorLinear(32 * 7 * 7, 100, seed=seed + 2), ReLU(), TensorLinear(100, 10, seed=seed + 3),
    )


class NanoGradRunner:
    def __init__(self, model: Sequential, lr: float) -> None:
        self.model = model
        self.optimizer = TensorAdam(model.parameters(), lr=lr)

    def input_gradient(self, images: list[float], labels: list[int]) -> tuple[list[float], list[float]]:
        """dL/dx for the mean cross entropy of the batch, and the logits."""

        x = Tensor(images, (len(labels), 1, 28, 28), requires_grad=True)
        logits = self.model(x)
        softmax_cross_entropy(logits, labels).backward()
        for parameter in self.model.parameters():    # only the input's gradient is wanted
            parameter.zero_grad()
        return list(x.grad), list(logits.data)

    def logits(self, images: list[float]) -> list[float]:
        with no_grad():
            return list(self.model(Tensor(images, (len(images) // PIXELS, 1, 28, 28))).data)

    def train_step(self, images: list[float], labels: list[int], lr: float) -> tuple[float, list[float]]:
        self.optimizer.lr = lr
        self.optimizer.zero_grad()
        logits = self.model(Tensor(images, (len(labels), 1, 28, 28)))
        loss = softmax_cross_entropy(logits, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item(), list(logits.data)

    def weights(self) -> list[list[float]]:
        return [list(p.data) for p in self.model.parameters()]


class TorchRunner:
    def __init__(self, model: Sequential, lr: float) -> None:
        import torch
        from torch_twin import TorchTwin

        self.torch = torch
        self.twin = TorchTwin(model)
        self.optimizer = torch.optim.Adam(self.twin.parameters, lr=lr)

    def _images(self, images):
        return self.torch.tensor(images, dtype=self.torch.float64).reshape(-1, 1, 28, 28)

    def input_gradient(self, images, labels):
        torch = self.torch
        x = self._images(images).requires_grad_()
        logits = self.twin(x)
        torch.nn.functional.cross_entropy(logits, torch.tensor(labels)).backward()
        for parameter in self.twin.parameters:
            parameter.grad = None
        return x.grad.reshape(-1).tolist(), logits.detach().reshape(-1).tolist()

    def logits(self, images):
        with self.torch.no_grad():
            return self.twin(self._images(images)).reshape(-1).tolist()

    def train_step(self, images, labels, lr: float):
        torch = self.torch
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad()
        logits = self.twin(self._images(images))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor(labels))
        loss.backward()
        self.optimizer.step()
        return loss.item(), logits.detach().reshape(-1).tolist()

    def weights(self):
        return [p.detach().reshape(-1).tolist() for p in self.twin.parameters]


# -- attacks ------------------------------------------------------------------


def predictions(logits: list[float]) -> list[int]:
    return [max(range(10), key=lambda c: logits[row * 10 + c]) for row in range(len(logits) // 10)]


def row_losses(logits: list[float], labels: list[int]) -> list[float]:
    """Cross entropy of each row: logsumexp(logits) - logits[label]."""

    losses = []
    for row, label in enumerate(labels):
        scores = logits[row * 10 : row * 10 + 10]
        top = max(scores)
        losses.append(top + math.log(sum(math.exp(s - top) for s in scores)) - scores[label])
    return losses


def add(images, delta) -> list[float]:
    return kernels.add(images, delta)


def fgsm(runner, images, labels, eps: float) -> list[float]:
    """The perturbation eps * sign(dL/dx), kept inside [0, 1]."""

    grad, _ = runner.input_gradient(images, labels)
    return kernels.projected_sign_step(kernels.zeros(len(images)), grad, images, eps, eps)


def pgd(runner, images, labels, *, eps: float, alpha: float, steps: int, restarts: int, rng: random.Random):
    """The strongest of ``restarts`` PGD runs per image (by final loss). As in
    Wong et al.'s code, an image stops moving once it is misclassified."""

    count = len(labels)
    best, best_loss = kernels.zeros(len(images)), [-1.0] * count
    no_grad_values = kernels.zeros(len(images))
    for _ in range(restarts):
        start = [rng.uniform(-eps, eps) for _ in range(len(images))]
        delta = kernels.projected_sign_step(start, no_grad_values, images, 0.0, eps)    # clip into the image
        for _ in range(steps):
            grad, logits = runner.input_gradient(add(images, delta), labels)
            still_right = [p == y for p, y in zip(predictions(logits), labels)]
            if not any(still_right):
                break
            stepped = kernels.projected_sign_step(delta, grad, images, alpha, eps)
            delta = kernels.storage(delta)
            for row in range(count):
                if still_right[row]:
                    delta[row * PIXELS : (row + 1) * PIXELS] = stepped[row * PIXELS : (row + 1) * PIXELS]
        losses = row_losses(runner.logits(add(images, delta)), labels)
        best = kernels.storage(best)
        for row in range(count):
            if losses[row] >= best_loss[row]:
                best_loss[row] = losses[row]
                best[row * PIXELS : (row + 1) * PIXELS] = delta[row * PIXELS : (row + 1) * PIXELS]
    return best


# -- training and evaluation --------------------------------------------------


def to_floats(images: list[bytes]) -> list[float]:
    return [pixel / 255.0 for image in images for pixel in image]


def train(runner, images, labels, args) -> dict:
    rng = random.Random(args.seed)
    order = list(range(len(labels)))
    batches = (len(order) + args.batch_size - 1) // args.batch_size
    start = perf_counter()
    for epoch in range(args.epochs):
        rng.shuffle(order)
        total = right = 0
        for index, first in enumerate(range(0, len(order), args.batch_size)):
            batch = order[first : first + args.batch_size]
            x, y = to_floats([images[i] for i in batch]), [labels[i] for i in batch]
            # The cyclic schedule: up to lr_max at 40% of training, back down to 0.
            t = epoch + (index + 1) / batches
            peak = args.epochs * 2 // 5
            lr = args.lr * (t / peak if t <= peak else (args.epochs - t) / (args.epochs - peak))
            if args.train == "fgsm":
                delta = [rng.uniform(-args.eps, args.eps) for _ in range(len(x))]
                grad, _ = runner.input_gradient(add(x, delta), y)
                x = add(x, kernels.projected_sign_step(delta, grad, x, args.alpha, args.eps))
            loss, logits = runner.train_step(x, y, lr)
            total += loss * len(y)
            right += sum(p == t for p, t in zip(predictions(logits), y))
        print(f"  epoch {epoch + 1:2d}: train loss {total / len(order):.4f}, "
              f"train accuracy {100 * right / len(order):.2f}% ({args.train} inputs), "
              f"{perf_counter() - start:.0f} s", flush=True)
    return {"seconds": perf_counter() - start}


def evaluate(runner, images, labels, args) -> dict:
    rng = random.Random(args.seed + 1)
    right = {"clean": 0, "fgsm": 0, "pgd": 0}
    for first in range(0, len(labels), args.batch_size):
        x, y = to_floats(images[first : first + args.batch_size]), labels[first : first + args.batch_size]
        right["clean"] += sum(p == t for p, t in zip(predictions(runner.logits(x)), y))
        right["fgsm"] += sum(p == t for p, t in zip(predictions(runner.logits(add(x, fgsm(runner, x, y, args.eps)))), y))
        delta = pgd(runner, x, y, eps=args.eps, alpha=args.pgd_alpha, steps=args.pgd_steps,
                    restarts=args.restarts, rng=rng)
        right["pgd"] += sum(p == t for p, t in zip(predictions(runner.logits(add(x, delta))), y))
    return {name: count / len(labels) for name, count in right.items()}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="data/mnist")
    parser.add_argument("--frameworks", nargs="+", choices=("nanograd", "torch"), default=["nanograd"])
    parser.add_argument("--train", choices=("none", "fgsm"), default="fgsm", help="adversarial training or not")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-3, help="peak of the cyclic schedule")
    parser.add_argument("--eps", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.375, help="FGSM training step")
    parser.add_argument("--pgd-alpha", type=float, default=0.01)
    parser.add_argument("--pgd-steps", type=int, default=50)
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    train_images, train_labels = load_split("mnist", Path(args.data), "train", args.train_limit)
    test_images, test_labels = load_split("mnist", Path(args.data), "test", args.test_limit)
    results, runners = {}, {}
    for framework in args.frameworks:
        model = small_cnn(seed=args.seed)
        if framework == "torch":
            import torch

            torch.set_num_threads(1)
            runners[framework] = runner = TorchRunner(model, args.lr)
        else:
            runners[framework] = runner = NanoGradRunner(model, args.lr)
        print(f"{framework}, {args.train} training", flush=True)
        trained = train(runner, train_images, train_labels, args)
        start = perf_counter()
        accuracy = evaluate(runner, test_images, test_labels, args)
        results[framework] = {**accuracy, "train_seconds": trained["seconds"], "attack_seconds": perf_counter() - start}
        print(f"{framework:8s}: clean {100 * accuracy['clean']:.2f}%, FGSM {100 * accuracy['fgsm']:.2f}%, "
              f"PGD-{args.pgd_steps} x {args.restarts} restarts {100 * accuracy['pgd']:.2f}% at eps {args.eps} "
              f"(training {trained['seconds']:.0f} s, attacks {results[framework]['attack_seconds']:.0f} s)",
              flush=True)
    if len(runners) == 2:
        difference = max(abs(a - b) for x, y in zip(runners["nanograd"].weights(), runners["torch"].weights())
                         for a, b in zip(x, y))
        print(f"largest weight difference after training: {difference:.1e}")
        results["max_weight_difference"] = difference
    return results


if __name__ == "__main__":
    main()
