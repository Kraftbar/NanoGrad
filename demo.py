"""Run a tiny scalar-autograd training demo."""

from __future__ import annotations

import random

from datasets import line_fitting, sign_separator, xor_gate
from engine import Value
from metrics import binary_accuracy
from model import MLP
from train import train_binary_classifier, train_mse


def regression_demo() -> None:
    xs, ys = line_fitting()
    model = MLP(inputs=1, layers=[1])
    history = train_mse(model, xs, ys, steps=80, lr=0.05)

    print("Regression demo")
    print(f"initial loss: {history[0]:.6f}")
    print(f"final loss:   {history[-1]:.6f}")


def classification_demo() -> None:
    xs, ys = sign_separator()
    model = MLP(inputs=1, layers=[1])
    history = train_binary_classifier(model, xs, ys, steps=80, lr=0.1)
    probabilities = binary_probabilities(model, xs)

    print("\nBinary classification demo")
    print(f"initial loss: {history[0]:.6f}")
    print(f"final loss:   {history[-1]:.6f}")
    print(f"accuracy:     {binary_accuracy(probabilities, ys):.3f}")


def xor_demo() -> None:
    xs, ys = xor_gate()
    model = MLP(inputs=2, layers=[4, 1])
    history = train_binary_classifier(model, xs, ys, steps=1000, lr=0.2)
    probabilities = binary_probabilities(model, xs)

    print("\nXOR demo")
    print(f"initial loss: {history[0]:.6f}")
    print(f"final loss:   {history[-1]:.6f}")
    print(f"accuracy:     {binary_accuracy(probabilities, ys):.3f}")

    for x, probability in zip(xs, probabilities):
        print(f"{x} -> {probability:.3f}")


def binary_probabilities(model: MLP, xs: list[list[float]]) -> list[float]:
    probabilities = []
    for x in xs:
        logit = model(x)
        if not isinstance(logit, Value):
            raise TypeError("binary demos expect the model to return one scalar Value")
        probabilities.append(logit.sigmoid().data)
    return probabilities


def main() -> None:
    random.seed(0)

    regression_demo()
    classification_demo()
    xor_demo()


if __name__ == "__main__":
    main()
