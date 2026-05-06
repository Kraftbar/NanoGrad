"""Tiny datasets used by demos and tests."""

from __future__ import annotations

import random
import gzip
import struct
from collections.abc import Iterator
from collections.abc import Sequence
from pathlib import Path


Feature = list
Batch = tuple[list[Feature], list[float]]
Sample = tuple[Feature, float]
CIFAR10_CHANNELS = 3
CIFAR10_ROWS = 32
CIFAR10_COLS = 32
CIFAR10_IMAGE_SIZE = CIFAR10_CHANNELS * CIFAR10_ROWS * CIFAR10_COLS
CIFAR10_RECORD_SIZE = 1 + CIFAR10_IMAGE_SIZE


class TinyDataset:
    """Small in-memory dataset for toy training checks."""

    def __init__(self, xs: list, ys: list[float]) -> None:
        if not xs:
            raise ValueError("dataset must not be empty")
        if len(xs) != len(ys):
            raise ValueError("features and targets must have the same length")

        self.feature_shape = _feature_shape(xs[0])
        if any(_feature_shape(sample) != self.feature_shape for sample in xs):
            raise ValueError("feature samples must have the same shape")

        self.xs = [
            _copy_feature(sample)
            for sample in xs
        ]
        self.ys = [float(value) for value in ys]

    def __len__(self) -> int:
        return len(self.ys)

    def __getitem__(self, index: int) -> Sample:
        return _copy_feature(self.xs[index]), self.ys[index]

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
                    _copy_feature(self.xs[index])
                    for index in batch_indices
                ],
                [
                    self.ys[index]
                    for index in batch_indices
                ],
            )


def make_batches(
    xs: list,
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


def _feature_shape(feature) -> tuple[int, ...]:
    if not _is_sequence(feature):
        raise ValueError("feature samples must be non-empty sequences")
    if not feature:
        raise ValueError("feature samples must not be empty")

    first = feature[0]
    if _is_sequence(first):
        child_shape = _feature_shape(first)
        for value in feature:
            if not _is_sequence(value) or _feature_shape(value) != child_shape:
                raise ValueError("feature samples must be rectangular")
        return (len(feature), *child_shape)

    if any(_is_sequence(value) for value in feature):
        raise ValueError("feature samples must be rectangular")
    return (len(feature),)


def _copy_feature(feature):
    if _is_sequence(feature):
        return [
            _copy_feature(value)
            for value in feature
        ]
    return float(feature)


def _is_sequence(value) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def load_mnist(
    images_path: str | Path,
    labels_path: str | Path,
    *,
    limit: int | None = None,
    channel_first: bool = False,
) -> TinyDataset:
    """Load local MNIST IDX image and label files into a TinyDataset."""

    return TinyDataset(
        read_mnist_images(
            images_path,
            limit=limit,
            channel_first=channel_first,
        ),
        read_mnist_labels(labels_path, limit=limit),
    )


def read_mnist_images(
    path: str | Path,
    *,
    limit: int | None = None,
    channel_first: bool = False,
) -> list:
    """Read local MNIST IDX image data as values in [0, 1]."""

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
            pixels = [pixel / 255.0 for pixel in raw]
            if channel_first:
                images.append([
                    [
                        pixels[row * cols : (row + 1) * cols]
                        for row in range(rows)
                    ],
                ])
            else:
                images.append(pixels)
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


def load_cifar10_batches(
    paths: Sequence[str | Path],
    *,
    limit: int | None = None,
    channel_first: bool = True,
) -> TinyDataset:
    """Load one or more local CIFAR-10 binary batch files."""

    xs, ys = read_cifar10_batches(
        paths,
        limit=limit,
        channel_first=channel_first,
    )
    return TinyDataset(xs, ys)


def read_cifar10_batches(
    paths: Sequence[str | Path],
    *,
    limit: int | None = None,
    channel_first: bool = True,
) -> tuple[list, list[float]]:
    """Read local CIFAR-10 binary batch files as values in [0, 1]."""

    if not paths:
        raise ValueError("CIFAR-10 paths must not be empty")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    images = []
    labels = []
    remaining = limit

    for path in paths:
        batch_images, batch_labels = read_cifar10_batch(
            path,
            limit=remaining,
            channel_first=channel_first,
        )
        images.extend(batch_images)
        labels.extend(batch_labels)

        if remaining is not None:
            remaining -= len(batch_labels)
            if remaining <= 0:
                break

    return images, labels


def read_cifar10_batch(
    path: str | Path,
    *,
    limit: int | None = None,
    channel_first: bool = True,
) -> tuple[list, list[float]]:
    """Read a local CIFAR-10 binary batch file."""

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    path = Path(path)
    size = path.stat().st_size
    if size % CIFAR10_RECORD_SIZE != 0:
        raise ValueError("CIFAR-10 binary batch has invalid byte length")

    count = _limited_count(size // CIFAR10_RECORD_SIZE, limit)
    images = []
    labels = []

    with path.open("rb") as file:
        for _ in range(count):
            record = file.read(CIFAR10_RECORD_SIZE)
            if len(record) != CIFAR10_RECORD_SIZE:
                raise ValueError("CIFAR-10 binary batch ended early")

            labels.append(float(record[0]))
            images.append(
                _cifar10_image(record[1:], channel_first=channel_first)
            )

    return images, labels


def _cifar10_image(raw: bytes, *, channel_first: bool) -> list:
    channels = [
        [
            pixel / 255.0
            for pixel in raw[start : start + CIFAR10_ROWS * CIFAR10_COLS]
        ]
        for start in range(0, CIFAR10_IMAGE_SIZE, CIFAR10_ROWS * CIFAR10_COLS)
    ]
    if channel_first:
        return [
            [
                channel[row * CIFAR10_COLS : (row + 1) * CIFAR10_COLS]
                for row in range(CIFAR10_ROWS)
            ]
            for channel in channels
        ]
    return [
        value
        for channel in channels
        for value in channel
    ]


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
