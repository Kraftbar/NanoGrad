"""Tiny datasets used by demos and tests."""

from __future__ import annotations

import random
from collections.abc import Iterator


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
