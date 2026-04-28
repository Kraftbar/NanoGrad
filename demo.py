"""Run a tiny scalar-autograd training demo."""

from __future__ import annotations

import random

from model import MLP
from train import train_binary_classifier, train_mse


def regression_demo() -> None:
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

    print("Regression demo")
    print(f"initial loss: {history[0]:.6f}")
    print(f"final loss:   {history[-1]:.6f}")


def classification_demo() -> None:
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

    model = MLP(inputs=1, layers=[1])
    history = train_binary_classifier(model, xs, ys, steps=80, lr=0.1)

    print("\nBinary classification demo")
    print(f"initial loss: {history[0]:.6f}")
    print(f"final loss:   {history[-1]:.6f}")


def main() -> None:
    random.seed(0)

    regression_demo()
    classification_demo()


if __name__ == "__main__":
    main()
