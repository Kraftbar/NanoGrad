"""Tiny datasets used by demos and tests."""

from __future__ import annotations

import random
import gzip
import struct
from collections.abc import Iterator
from pathlib import Path


Batch = tuple[list[list[float]], list[float]]
Sample = tuple[list[float], float]


class TinyDataset:
    """Small in-memory dataset for toy training checks."""

    def __init__(self, xs: list[list[float]], ys: list[float]) -> None:
        if not xs:
            raise ValueError("dataset must not be empty")
        if len(xs) != len(ys):
            raise ValueError("features and targets must have the same length")

        width = len(xs[0])
        if width == 0:
            raise ValueError("feature rows must not be empty")
        if any(len(row) != width for row in xs):
            raise ValueError("feature rows must have the same length")

        self.xs = [
            [float(value) for value in row]
            for row in xs
        ]
        self.ys = [float(value) for value in ys]

    def __len__(self) -> int:
        return len(self.ys)

    def __getitem__(self, index: int) -> Sample:
        return self.xs[index][:], self.ys[index]

    def batches(
        self,
        batch_size: int,
        *,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> Iterator[Batch]:
        """Yield feature and target batches."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        indices = list(range(len(self)))
        if shuffle:
            random.Random(seed).shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            yield (
                [
                    self.xs[index][:]
                    for index in batch_indices
                ],
                [
                    self.ys[index]
                    for index in batch_indices
                ],
            )


def make_batches(
    xs: list[list[float]],
    ys: list[float],
    batch_size: int,
    *,
    shuffle: bool = False,
    seed: int | None = None,
) -> Iterator[Batch]:
    """Create batches from raw feature and target lists."""

    return TinyDataset(xs, ys).batches(
        batch_size,
        shuffle=shuffle,
        seed=seed,
    )


def load_mnist(
    images_path: str | Path,
    labels_path: str | Path,
    *,
    limit: int | None = None,
) -> TinyDataset:
    """Load local MNIST IDX image and label files into a TinyDataset."""

    return TinyDataset(
        read_mnist_images(images_path, limit=limit),
        read_mnist_labels(labels_path, limit=limit),
    )


def read_mnist_images(
    path: str | Path,
    *,
    limit: int | None = None,
) -> list[list[float]]:
    """Read local MNIST IDX image data as flattened values in [0, 1]."""

    with _open_idx(path) as file:
        magic, count, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError("MNIST image file has invalid magic number")

        count = _limited_count(count, limit)
        image_size = rows * cols
        images = []
        for _ in range(count):
            raw = file.read(image_size)
            if len(raw) != image_size:
                raise ValueError("MNIST image file ended early")
            images.append([pixel / 255.0 for pixel in raw])
        return images


def read_mnist_labels(
    path: str | Path,
    *,
    limit: int | None = None,
) -> list[float]:
    """Read local MNIST IDX label data."""

    with _open_idx(path) as file:
        magic, count = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError("MNIST label file has invalid magic number")

        count = _limited_count(count, limit)
        raw = file.read(count)
        if len(raw) != count:
            raise ValueError("MNIST label file ended early")
        return [float(label) for label in raw]


def line_fitting() -> tuple[list[list[float]], list[float]]:
    """Small scalar regression dataset for learning y = 3x + 1."""

    xs = [
        [-1.0],
        [0.0],
        [1.0],
        [2.0],
    ]
    ys = [
        -2.0,
        1.0,
        4.0,
        7.0,
    ]
    return xs, ys


def noisy_line_fitting() -> tuple[list[list[float]], list[float]]:
    """Small regression dataset near y = 3x + 1 with fixed noise."""

    xs = [
        [-2.0],
        [-1.0],
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    ]
    ys = [
        -5.2,
        -1.8,
        1.1,
        3.9,
        7.2,
        9.8,
    ]
    return xs, ys


def sign_separator() -> tuple[list[list[float]], list[float]]:
    """Binary dataset for separating negative and positive scalar inputs."""

    xs = [
        [-2.0],
        [-1.0],
        [1.0],
        [2.0],
    ]
    ys = [
        0.0,
        0.0,
        1.0,
        1.0,
    ]
    return xs, ys


def tiny_2d_clusters() -> tuple[list[list[float]], list[float]]:
    """Tiny linearly separable 2D cluster dataset."""

    xs = [
        [-2.0, -1.0],
        [-1.5, -2.0],
        [-1.0, -1.0],
        [1.0, 1.5],
        [1.5, 1.0],
        [2.0, 2.0],
    ]
    ys = [
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
    ]
    return xs, ys


def and_gate() -> tuple[list[list[float]], list[float]]:
    """AND logic gate dataset for linear binary classification."""

    xs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    ys = [
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    return xs, ys


def or_gate() -> tuple[list[list[float]], list[float]]:
    """OR logic gate dataset for linear binary classification."""

    xs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    ys = [
        0.0,
        1.0,
        1.0,
        1.0,
    ]
    return xs, ys


def xor_gate() -> tuple[list[list[float]], list[float]]:
    """XOR logic gate dataset for non-linear binary classification."""

    xs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    ys = [
        0.0,
        1.0,
        1.0,
        0.0,
    ]
    return xs, ys


def _open_idx(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def _limited_count(count: int, limit: int | None) -> int:
    if limit is None:
        return count
    if limit < 0:
        raise ValueError("limit must be non-negative")
    return min(count, limit)
