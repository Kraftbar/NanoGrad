"""Train the same LeNet-style CNN on MNIST in NanoGrad and in PyTorch.

The model is the classic LeNet-5 shape, scaled to 28x28 inputs:

    conv 6@5x5 -> tanh -> avgpool 2 -> conv 16@5x5 -> tanh -> avgpool 2
    -> flatten (16*4*4 = 256) -> linear 10

PyTorch starts from NanoGrad's exact initial weights and sees the same
batches in the same order with the same Adam settings. In float64 the two
runs should then agree to many decimal places, so this is a correctness
check of NanoGrad's convolution, pooling, autograd and optimizer, and a
speed comparison on the same work.

PyTorch is only needed for the comparison side:

    python3 -m venv .venv && .venv/bin/pip install torch
    .venv/bin/python mnist_compare.py --epochs 1

    python3 mnist_compare.py --frameworks nanograd     # NanoGrad alone
"""

from __future__ import annotations

import argparse
import importlib.util
import random
from pathlib import Path
from time import perf_counter

from datasets import read_mnist_images, read_mnist_labels
from losses import softmax_cross_entropy
from optim import TensorAdam
from tensor import Tensor, no_grad
from train import evaluation_mode
from vision import TwoConvCNN


FRAMEWORKS = ("nanograd", "torch")
IMAGE_SHAPE = (1, 28, 28)
IMAGE_SIZE = 28 * 28


# ---------------------------------------------------------------------------
# Data: flat 784-value images and one fixed batch order shared by both sides
# ---------------------------------------------------------------------------


def load_split(data_dir: Path, prefix: str, limit: int | None) -> tuple[list[list[float]], list[int]]:
    images = read_mnist_images(data_dir / f"{prefix}-images-idx3-ubyte.gz", limit=limit)
    labels = read_mnist_labels(data_dir / f"{prefix}-labels-idx1-ubyte.gz", limit=limit)
    return [list(image) for image in images], [int(label) for label in labels]


def epoch_batches(count: int, batch_size: int, rng: random.Random) -> list[list[int]]:
    """A shuffled split of ``range(count)`` into batches (the last may be short)."""

    order = list(range(count))
    rng.shuffle(order)
    return [order[start : start + batch_size] for start in range(0, count, batch_size)]


def batch_pixels(images: list[list[float]], indices: list[int]) -> list[float]:
    return [value for index in indices for value in images[index]]


# ---------------------------------------------------------------------------
# NanoGrad side
# ---------------------------------------------------------------------------


def build_nanograd(seed: int) -> TwoConvCNN:
    return TwoConvCNN(image_shape=IMAGE_SHAPE, classes=10, pooling="avg", activation="tanh", seed=seed)


class NanoGradRunner:
    def __init__(self, model: TwoConvCNN, lr: float) -> None:
        self.model = model
        self.optimizer = TensorAdam(model.parameters(), lr=lr)

    def train_step(self, pixels: list[float], labels: list[int]) -> float:
        inputs = Tensor(pixels, (len(labels), *IMAGE_SHAPE))
        self.optimizer.zero_grad()
        loss = softmax_cross_entropy(self.model(inputs), labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, pixels: list[float], count: int) -> list[int]:
        with no_grad(), evaluation_mode(self.model):
            logits = self.model(Tensor(pixels, (count, *IMAGE_SHAPE)))
        return [
            max(range(10), key=lambda k, row=row: logits.data[row * 10 + k])
            for row in range(count)
        ]

    def weights(self) -> list[list[float]]:
        return [parameter.data for parameter in self.model.parameters()]


# ---------------------------------------------------------------------------
# PyTorch side: the identical network, loaded with NanoGrad's weights
# ---------------------------------------------------------------------------


class TorchRunner:
    def __init__(self, model: TwoConvCNN, lr: float, dtype: str, threads: int) -> None:
        import torch
        from torch import nn

        torch.set_num_threads(threads)
        self.torch = torch
        self.dtype = getattr(torch, dtype)
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, 5), nn.Tanh(), nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2),
            nn.Flatten(), nn.Linear(256, 10),
        ).to(self.dtype)
        # NanoGrad and PyTorch share layouts: kernels (F, C, k, k), linear
        # weights (out, in), and row-major flatten. Biases just need reshaping.
        torch_parameters = [
            self.net[0].weight, self.net[0].bias,
            self.net[3].weight, self.net[3].bias,
            self.net[7].weight, self.net[7].bias,
        ]
        with torch.no_grad():
            for target, source in zip(torch_parameters, model.parameters()):
                target.copy_(torch.tensor(source.data, dtype=self.dtype).reshape(target.shape))
        self.parameters = torch_parameters
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def _inputs(self, pixels: list[float], count: int):
        return self.torch.tensor(pixels, dtype=self.dtype).reshape(count, *IMAGE_SHAPE)

    def train_step(self, pixels: list[float], labels: list[int]) -> float:
        inputs = self._inputs(pixels, len(labels))
        self.optimizer.zero_grad()
        loss = self.torch.nn.functional.cross_entropy(self.net(inputs), self.torch.tensor(labels))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, pixels: list[float], count: int) -> list[int]:
        with self.torch.no_grad():
            return self.net(self._inputs(pixels, count)).argmax(dim=1).tolist()

    def weights(self) -> list[list[float]]:
        return [parameter.detach().reshape(-1).tolist() for parameter in self.parameters]


# ---------------------------------------------------------------------------
# Training loop shared by both runners
# ---------------------------------------------------------------------------


def accuracy(runner, images: list[list[float]], labels: list[int], batch_size: int = 500) -> float:
    correct = 0
    for start in range(0, len(labels), batch_size):
        indices = list(range(start, min(start + batch_size, len(labels))))
        predictions = runner.predict(batch_pixels(images, indices), len(indices))
        correct += sum(prediction == labels[index] for prediction, index in zip(predictions, indices))
    return correct / len(labels)


def run(args: argparse.Namespace) -> dict:
    data_dir = Path(args.data_dir)
    train_images, train_labels = load_split(data_dir, "train", args.train_limit)
    test_images, test_labels = load_split(data_dir, "t10k", args.test_limit)
    print(f"train {len(train_labels)} images, test {len(test_labels)} images, batch {args.batch_size}")

    model = build_nanograd(args.seed)
    runners = {}
    for name in args.frameworks:
        if name == "nanograd":
            runners[name] = NanoGradRunner(model, args.lr)
        else:
            if importlib.util.find_spec("torch") is None:
                raise SystemExit("PyTorch is not installed; use --frameworks nanograd")
            # Built before NanoGrad trains, so it copies the initial weights.
            runners[name] = TorchRunner(model, args.lr, args.torch_dtype, args.torch_threads)

    rng = random.Random(args.seed)
    schedule = [epoch_batches(len(train_labels), args.batch_size, rng) for _ in range(args.epochs)]

    results = {name: [] for name in runners}
    for name, runner in runners.items():
        for epoch, batches in enumerate(schedule, start=1):
            start = perf_counter()
            losses = [
                runner.train_step(batch_pixels(train_images, batch), [train_labels[i] for i in batch])
                for batch in batches
            ]
            train_seconds = perf_counter() - start
            test_accuracy = accuracy(runner, test_images, test_labels)
            row = {
                "epoch": epoch,
                "train_loss": sum(losses) / len(losses),
                "test_accuracy": test_accuracy,
                "train_seconds": train_seconds,
                "ms_per_step": 1000.0 * train_seconds / len(batches),
            }
            results[name].append(row)
            print(f"{name:8s} epoch {epoch}: loss {row['train_loss']:.4f}, "
                  f"test acc {100 * test_accuracy:.2f}%, {train_seconds:.1f} s "
                  f"({row['ms_per_step']:.1f} ms/step)", flush=True)

    if len(runners) == 2:
        difference = max(
            abs(a - b)
            for left, right in zip(runners["nanograd"].weights(), runners["torch"].weights())
            for a, b in zip(left, right)
        )
        print(f"max |weight difference| after training: {difference:.3e}")
        results["max_weight_difference"] = difference
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", default="data/mnist")
    parser.add_argument("--frameworks", nargs="+", choices=FRAMEWORKS, default=list(FRAMEWORKS))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
