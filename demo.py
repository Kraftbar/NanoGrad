"""Run a tiny scalar-autograd training demo."""

from __future__ import annotations

import random

from model import MLP
from train import train_mse


def main() -> None:
    random.seed(0)

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

    model = MLP(inputs=1, layers=[1])
    history = train_mse(model, xs, ys, steps=80, lr=0.05)

    print(f"initial loss: {history[0]:.6f}")
    print(f"final loss:   {history[-1]:.6f}")


if __name__ == "__main__":
    main()
