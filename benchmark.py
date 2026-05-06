"""Small core-operation timing checks for NanoGrad."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from time import perf_counter

from tensor import Tensor, conv2d_valid, matmul


def benchmark_matmul() -> None:
    left = Tensor(
        _values(32 * 64),
        (32, 64),
        requires_grad=True,
    )
    right = Tensor(
        _values(64 * 32, offset=5),
        (64, 32),
        requires_grad=True,
    )

    loss = matmul(left, right).mean()
    loss.backward()


def benchmark_conv2d() -> None:
    image = Tensor(
        _values(4 * 1 * 12 * 12),
        (4, 1, 12, 12),
        requires_grad=True,
    )
    kernel = Tensor(
        _values(6 * 1 * 3 * 3, offset=11),
        (6, 1, 3, 3),
        requires_grad=True,
    )

    loss = conv2d_valid(image, kernel).mean()
    loss.backward()


def run(args: argparse.Namespace) -> None:
    benchmarks = [
        ("matmul fwd+bwd", benchmark_matmul),
        ("conv2d fwd+bwd", benchmark_conv2d),
    ]

    print("NanoGrad core benchmarks")
    print(f"repeat: {args.repeat}")
    for name, function in benchmarks:
        seconds = average_seconds(function, repeat=args.repeat)
        print(f"{name}: {seconds:.6f}s")


def average_seconds(function: Callable[[], None], *, repeat: int) -> float:
    if repeat <= 0:
        raise ValueError("repeat must be positive")

    start = perf_counter()
    for _ in range(repeat):
        function()
    return (perf_counter() - start) / repeat


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3)
    return parser.parse_args(argv)


def _values(count: int, *, offset: int = 0) -> list[float]:
    return [
        ((index + offset) % 17 - 8) / 10
        for index in range(count)
    ]


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
