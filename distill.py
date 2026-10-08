"""Knowledge distillation on MNIST (Hinton, Vinyals and Dean, 2015), NanoGrad vs PyTorch.

A big, heavily regularized network generalizes better than a small one
trained alone. Distillation moves that knowledge into the small network:
train the small "student" to match the big "teacher's" predicted
probabilities, softened by a temperature T,

    q = softmax(teacher logits / T)          soft targets: how much a 7 looks like a 1
    loss = alpha T^2 CE(q, softmax(student logits / T)) + (1 - alpha) CE(labels, student logits)

At T = 1 the teacher is almost always ~100% sure of one class and its
targets say little more than the label. At T = 20 the small
probabilities of the wrong classes become visible, and they carry what
the teacher learned about which digits resemble each other. The T^2 keeps
the soft term's gradients on the same scale as T changes.

The paper's MNIST experiment: a teacher of two 1,200-unit ReLU layers
with dropout and inputs jittered by up to 2 pixels makes 67 test errors.
A student of two 800-unit layers without regularization makes 146 alone
and 74 when distilled at T = 20. Here the training is shorter (Adam,
lr 1e-3, batch 100), so the counts differ, but the comparison is the same.

    python3 distill.py --epochs 20
    .venv/bin/python distill.py --compare        # one distillation epoch in both frameworks

Data: data/mnist/.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from time import perf_counter

from losses import soft_cross_entropy, softmax_cross_entropy
from mnist_compare import load_split
from optim import TensorAdam
from tensor import Tensor, no_grad
from tensor_nn import ReLU, Sequential, TensorDropout, TensorLinear


def mlp(sizes: list[int], *, dropout: tuple[float, float] = (0.0, 0.0), seed: int = 0) -> Sequential:
    """Linear-ReLU layers; dropout[0] on the input, dropout[1] on each hidden layer."""

    layers = [TensorDropout(dropout[0], seed=seed + 100)] if dropout[0] else []
    for index, (a, b) in enumerate(zip(sizes, sizes[1:])):
        layers.append(TensorLinear(a, b, seed=seed + index))
        if index < len(sizes) - 2:
            layers.append(ReLU())
            if dropout[1]:
                layers.append(TensorDropout(dropout[1], seed=seed + 200 + index))
    return Sequential(*layers)


def jitter(image: bytes, rng: random.Random, pixels: int = 2) -> list[float]:
    """Shift a 28x28 image by up to ``pixels`` in each direction (zeros come in)."""

    dy, dx = rng.randint(-pixels, pixels), rng.randint(-pixels, pixels)
    return [image[(y - dy) * 28 + (x - dx)] / 255.0 if 0 <= y - dy < 28 and 0 <= x - dx < 28 else 0.0
            for y in range(28) for x in range(28)]


def pixels(images: list[bytes]) -> list[float]:
    return [p / 255.0 for image in images for p in image]


def softened(logits: list[float], temperature: float) -> list[float]:
    """softmax(logits / T), row by row (rows of 10)."""

    import math

    out = []
    for row in range(len(logits) // 10):
        values = [v / temperature for v in logits[row * 10 : row * 10 + 10]]
        top = max(values)
        weights = [math.exp(v - top) for v in values]
        total = sum(weights)
        out += [w / total for w in weights]
    return out


def distillation_loss(student_logits: Tensor, soft_targets: Tensor | None, labels: list[int], *,
                      temperature: float, alpha: float) -> Tensor:
    hard = softmax_cross_entropy(student_logits, labels)
    if soft_targets is None:
        return hard
    soft = soft_cross_entropy(student_logits * (1.0 / temperature), soft_targets)
    return soft * (alpha * temperature ** 2) + hard * (1.0 - alpha)


def train(model: Sequential, images, labels, args, *, augment: bool = False, soft: list[float] | None = None,
          label: str = "") -> float:
    """Adam on the training set; ``soft`` holds the teacher's soft targets (10 per image)."""

    optimizer = TensorAdam(model.parameters(), lr=args.lr)
    rng, order, start = random.Random(args.seed), list(range(len(labels))), perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(order)
        total = 0.0
        for first in range(0, len(order), args.batch_size):
            batch = order[first : first + args.batch_size]
            x = [v for i in batch for v in jitter(images[i], rng)] if augment else pixels([images[i] for i in batch])
            targets = None if soft is None else Tensor([v for i in batch for v in soft[i * 10 : i * 10 + 10]],
                                                       (len(batch), 10))
            optimizer.zero_grad()
            loss = distillation_loss(model(Tensor(x, (len(batch), 784))), targets, [labels[i] for i in batch],
                                     temperature=args.temperature, alpha=args.alpha)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        print(f"  {label} epoch {epoch:2d}: loss {total / len(order):.4f}, {perf_counter() - start:.0f} s", flush=True)
    return perf_counter() - start


def logits_of(model: Sequential, images: list[bytes]) -> list[float]:
    model.eval()
    out = []
    with no_grad():
        for first in range(0, len(images), 1000):
            chunk = images[first : first + 1000]
            out += list(model(Tensor(pixels(chunk), (len(chunk), 784))).data)
    return out


def errors(model: Sequential, images, labels) -> int:
    scores = logits_of(model, images)
    return sum(max(range(10), key=lambda c: scores[i * 10 + c]) != label for i, label in enumerate(labels))


def compare(args, train_images, train_labels) -> dict:
    """One distillation epoch of the student, NanoGrad and PyTorch from the same weights,
    on soft targets from an untrained (but fixed) teacher."""

    import torch

    from torch_twin import TorchTwin

    torch.set_num_threads(1)
    teacher = mlp([784, 64, 10], seed=args.seed + 7)
    soft = softened(logits_of(teacher, train_images), args.temperature)
    student = mlp([784, 100, 10], seed=args.seed)
    twin = TorchTwin(student)
    optimizer = torch.optim.Adam(twin.parameters, lr=args.lr)
    start = perf_counter()
    train(student, train_images, train_labels, args, soft=soft, label="nanograd")
    seconds = {"nanograd": perf_counter() - start}
    rng, order, start = random.Random(args.seed), list(range(len(train_labels))), perf_counter()
    rng.shuffle(order)
    for first in range(0, len(order), args.batch_size):
        batch = order[first : first + args.batch_size]
        x = torch.tensor(pixels([train_images[i] for i in batch]), dtype=torch.float64).reshape(-1, 784)
        q = torch.tensor([v for i in batch for v in soft[i * 10 : i * 10 + 10]], dtype=torch.float64).reshape(-1, 10)
        optimizer.zero_grad()
        logits = twin(x)
        soft_loss = -(q * torch.log_softmax(logits / args.temperature, dim=1)).sum(1).mean()
        hard_loss = torch.nn.functional.cross_entropy(logits, torch.tensor([train_labels[i] for i in batch]))
        (args.alpha * args.temperature ** 2 * soft_loss + (1 - args.alpha) * hard_loss).backward()
        optimizer.step()
    seconds["torch"] = perf_counter() - start
    difference = max(abs(a - b) for p, t in zip(student.parameters(), twin.parameters)
                     for a, b in zip(p.data, t.detach().reshape(-1).tolist()))
    print(f"largest weight difference after one epoch: {difference:.1e} "
          f"(NanoGrad {seconds['nanograd']:.1f} s, PyTorch {seconds['torch']:.1f} s)")
    return {"max_weight_difference": difference, "seconds": seconds}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="data/mnist")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=20.0)
    parser.add_argument("--alpha", type=float, default=0.9, help="weight of the soft targets")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    train_images, train_labels = load_split("mnist", Path(args.data), "train", args.train_limit)
    test_images, test_labels = load_split("mnist", Path(args.data), "test", args.test_limit)
    if args.compare:
        args.epochs = 1
        return compare(args, train_images, train_labels)

    teacher = mlp([784, 1200, 1200, 10], dropout=(0.2, 0.5), seed=args.seed + 10)
    train(teacher, train_images, train_labels, args, augment=True, label="teacher")
    results = {"teacher": errors(teacher, test_images, test_labels)}
    print(f"teacher (2 x 1200, dropout, jitter): {results['teacher']} test errors", flush=True)

    alone = mlp([784, 800, 800, 10], seed=args.seed)
    train(alone, train_images, train_labels, args, label="student alone")
    results["student"] = errors(alone, test_images, test_labels)
    print(f"student alone (2 x 800): {results['student']} test errors", flush=True)

    soft = softened(logits_of(teacher, train_images), args.temperature)
    distilled = mlp([784, 800, 800, 10], seed=args.seed)               # the same starting weights
    train(distilled, train_images, train_labels, args, soft=soft, label="student distilled")
    results["distilled"] = errors(distilled, test_images, test_labels)
    print(f"student distilled at T = {args.temperature:g}: {results['distilled']} test errors "
          f"(of {len(test_labels)})", flush=True)
    return results


if __name__ == "__main__":
    main()
