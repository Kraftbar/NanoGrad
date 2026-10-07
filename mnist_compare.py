"""Train the same image classifier in NanoGrad and in PyTorch.

Datasets (``--dataset``): MNIST digits, Fashion-MNIST clothing (same format,
harder), and CIFAR-10 colour photos (3x32x32, much harder). See the README
in each ``data/`` folder for where to get the files.

Three model families, all built on the same NanoGrad core:

- ``lenet``: the classic LeNet-5 shape, scaled to 28x28 inputs:
  conv 6@5x5 -> tanh -> avgpool 2 -> conv 16@5x5 -> tanh -> avgpool 2
  -> flatten (16*4*4 = 256) -> linear 10
- ``resnet``: a small residual network with batch norm (resnet.py)
- ``vit``: a tiny vision transformer on 7x7 patches (vit.py)

PyTorch runs the same architecture (torch_twin.py), starts from NanoGrad's
exact initial weights, and sees the same batches in the same order with the
same Adam settings. In float64 the two runs should then agree to many
decimal places, so this is a correctness check of NanoGrad's layers,
autograd and optimizer, and a speed comparison on the same work.

PyTorch is only needed for the comparison side:

    python3 -m venv .venv && .venv/bin/pip install torch
    .venv/bin/python mnist_compare.py --epochs 1 --model resnet
    .venv/bin/python mnist_compare.py --dataset cifar10 --model resnet

    python3 mnist_compare.py --frameworks nanograd     # NanoGrad alone
"""
from __future__ import annotations

import argparse
import importlib.util
import random
from pathlib import Path
from time import perf_counter

import gzip
import struct
from functools import partial

from losses import softmax_cross_entropy
from optim import TensorAdam
from parallel import DataParallel
from resnet import SmallResNet
from tensor import Tensor, no_grad
from tensor_nn import TensorModule
from train import evaluation_mode
from vision import TwoConvCNN
from vit import TinyViT


FRAMEWORKS = ("nanograd", "torch")
IMAGE_SHAPE = (1, 28, 28)
IMAGE_SIZE = 28 * 28

DATASETS = {
    # name: (default directory, image shape)
    "mnist": ("data/mnist", (1, 28, 28)),
    "fashion": ("data/fashion_mnist", (1, 28, 28)),
    "cifar10": ("data/cifar10/cifar-10-batches-bin", (3, 32, 32)),
}


# ---------------------------------------------------------------------------
# Data: images stay as raw bytes (one per pixel value) until a batch needs
# them, so all of CIFAR-10 fits in memory; one fixed batch order is shared
# by both frameworks
# ---------------------------------------------------------------------------


_PIXEL = [value / 255.0 for value in range(256)]    # byte -> value in [0, 1]


def load_split(dataset: str, data_dir: Path, split: str, limit: int | None) -> tuple[list[bytes], list[int]]:
    """(images as bytes, labels) for ``split`` "train" or "test"."""

    if dataset == "cifar10":
        names = [f"data_batch_{n}.bin" for n in range(1, 6)] if split == "train" else ["test_batch.bin"]
        record = 1 + 3 * 32 * 32                         # label byte, then the pixels
        images, labels = [], []
        for name in names:
            raw = (data_dir / name).read_bytes()
            for start in range(0, len(raw), record):
                labels.append(raw[start])
                images.append(raw[start + 1 : start + record])
        return images[:limit], labels[:limit]

    prefix = "train" if split == "train" else "t10k"
    with gzip.open(data_dir / f"{prefix}-images-idx3-ubyte.gz", "rb") as file:
        _, count, rows, cols = struct.unpack(">IIII", file.read(16))
        raw = file.read()
    with gzip.open(data_dir / f"{prefix}-labels-idx1-ubyte.gz", "rb") as file:
        labels = list(file.read()[8:])
    size = rows * cols
    images = [raw[i * size : (i + 1) * size] for i in range(count)]
    return images[:limit], labels[:limit]


def epoch_batches(count: int, batch_size: int, rng: random.Random) -> list[list[int]]:
    """A shuffled split of ``range(count)`` into batches (the last may be short)."""

    order = list(range(count))
    rng.shuffle(order)
    return [order[start : start + batch_size] for start in range(0, count, batch_size)]


def batch_pixels(images: list[bytes], indices: list[int]) -> list[float]:
    return [_PIXEL[byte] for index in indices for byte in images[index]]


# ---------------------------------------------------------------------------
# NanoGrad side
# ---------------------------------------------------------------------------


MODELS = ("lenet", "resnet", "vit")


def build_nanograd(seed: int, model: str = "lenet", image_shape: tuple[int, int, int] = IMAGE_SHAPE) -> TensorModule:
    """The three model families: a classic CNN, a ResNet and a ViT."""

    if model == "lenet":
        return TwoConvCNN(image_shape=image_shape, classes=10, pooling="avg", activation="tanh", seed=seed)
    if model == "resnet":
        return SmallResNet(image_shape=image_shape, classes=10, seed=seed)
    if model == "vit":
        # A 4x4 grid of patches: 7x7 on MNIST, 8x8 on CIFAR-10.
        return TinyViT(image_shape=image_shape, patch_size=image_shape[1] // 4, classes=10, seed=seed)
    raise ValueError(f"unknown model: {model}")


def image_loss(model: TensorModule, xs: list[list[float]], ys: list[int], image_shape=IMAGE_SHAPE):
    """Mean cross entropy of a batch of flat images (also run by workers)."""

    inputs = Tensor([value for image in xs for value in image], (len(xs), *image_shape))
    return softmax_cross_entropy(model(inputs), ys)


class NanoGradRunner:
    def __init__(self, model: TensorModule, lr: float, workers: int = 1, image_shape=IMAGE_SHAPE) -> None:
        self.model = model
        self.image_shape = tuple(image_shape)
        self.image_size = image_shape[0] * image_shape[1] * image_shape[2]
        self.optimizer = TensorAdam(model.parameters(), lr=lr)
        # With workers > 1 each batch is split across processes (parallel.py).
        loss = partial(image_loss, image_shape=self.image_shape)
        self.parallel = DataParallel(model, loss, workers=workers) if workers > 1 else None

    def train_step(self, pixels: list[float], labels: list[int]) -> float:
        self.optimizer.zero_grad()
        if self.parallel is not None:
            size = self.image_size
            images = [pixels[i * size : (i + 1) * size] for i in range(len(labels))]
            loss = self.parallel.loss_and_gradients(images, labels)
        else:
            inputs = Tensor(pixels, (len(labels), *self.image_shape))
            loss_tensor = softmax_cross_entropy(self.model(inputs), labels)
            loss_tensor.backward()
            loss = loss_tensor.item()
        self.optimizer.step()
        return loss

    def close(self) -> None:
        if self.parallel is not None:
            self.parallel.close()

    def predict(self, pixels: list[float], count: int) -> list[int]:
        with no_grad(), evaluation_mode(self.model):
            logits = self.model(Tensor(pixels, (count, *self.image_shape)))
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
    """The same model in PyTorch (see torch_twin.py), with PyTorch's Adam."""

    def __init__(self, model: TensorModule, lr: float, dtype: str, threads: int, image_shape=IMAGE_SHAPE) -> None:
        import torch

        from torch_twin import TorchTwin

        torch.set_num_threads(threads)
        self.torch = torch
        self.dtype = getattr(torch, dtype)
        self.image_shape = tuple(image_shape)
        self.twin = TorchTwin(model, dtype=self.dtype)
        self.optimizer = torch.optim.Adam(self.twin.parameters, lr=lr)

    def _inputs(self, pixels: list[float], count: int):
        return self.torch.tensor(pixels, dtype=self.dtype).reshape(count, *self.image_shape)

    def train_step(self, pixels: list[float], labels: list[int]) -> float:
        inputs = self._inputs(pixels, len(labels))
        self.twin.training = True
        self.optimizer.zero_grad()
        loss = self.torch.nn.functional.cross_entropy(self.twin(inputs), self.torch.tensor(labels))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, pixels: list[float], count: int) -> list[int]:
        self.twin.training = False
        with self.torch.no_grad():
            return self.twin(self._inputs(pixels, count)).argmax(dim=1).tolist()

    def weights(self) -> list[list[float]]:
        return [parameter.detach().reshape(-1).tolist() for parameter in self.twin.parameters]


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
    default_dir, image_shape = DATASETS[args.dataset]
    data_dir = Path(args.data_dir or default_dir)
    train_images, train_labels = load_split(args.dataset, data_dir, "train", args.train_limit)
    test_images, test_labels = load_split(args.dataset, data_dir, "test", args.test_limit)
    print(f"{args.dataset}: train {len(train_labels)} images, test {len(test_labels)} images, "
          f"batch {args.batch_size}")

    model = build_nanograd(args.seed, args.model, image_shape)
    print(f"model {args.model}, {model.num_parameters()} parameters")
    runners = {}
    for name in args.frameworks:
        if name == "nanograd":
            runners[name] = NanoGradRunner(model, args.lr, workers=args.workers, image_shape=image_shape)
        else:
            if importlib.util.find_spec("torch") is None:
                raise SystemExit("PyTorch is not installed; use --frameworks nanograd")
            # Built before NanoGrad trains, so it copies the initial weights.
            runners[name] = TorchRunner(model, args.lr, args.torch_dtype, args.torch_threads, image_shape)

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

    if "nanograd" in runners:
        runners["nanograd"].close()
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
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="mnist")
    parser.add_argument("--data-dir", help="defaults to the dataset's folder under data/")
    parser.add_argument("--frameworks", nargs="+", choices=FRAMEWORKS, default=list(FRAMEWORKS))
    parser.add_argument("--model", choices=MODELS, default="lenet")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1, help="NanoGrad worker processes")
    parser.add_argument("--torch-dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
