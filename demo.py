"""Run a tiny scalar-autograd training demo."""

from __future__ import annotations

import random

from datasets import (
    and_gate,
    line_fitting,
    noisy_line_fitting,
    or_gate,
    sign_separator,
    tiny_2d_clusters,
    xor_gate,
)
from model import MLP
from train import (
    binary_probabilities,
    train_binary_classifier_summary,
    train_mse_summary,
)


def regression_demo() -> None:
    xs, ys = line_fitting()
    model = MLP(inputs=1, layers=[1])
    summary = train_mse_summary(model, xs, ys, steps=80, lr=0.05)

    print("Regression demo")
    print_summary(summary)


def noisy_regression_demo() -> None:
    xs, ys = noisy_line_fitting()
    model = MLP(inputs=1, layers=[1])
    summary = train_mse_summary(model, xs, ys, steps=80, lr=0.03)

    print("\nNoisy regression demo")
    print_summary(summary)


def classification_demo() -> None:
    xs, ys = sign_separator()
    model = MLP(inputs=1, layers=[1])
    summary = train_binary_classifier_summary(model, xs, ys, steps=80, lr=0.1)

    print("\nBinary classification demo")
    print_summary(summary)


def cluster_demo() -> None:
    xs, ys = tiny_2d_clusters()
    model = MLP(inputs=2, layers=[1])
    summary = train_binary_classifier_summary(model, xs, ys, steps=80, lr=0.1)

    print("\nTiny 2D cluster demo")
    print_summary(summary)


def logic_gate_demo() -> None:
    print("\nLogic gate demos")

    for name, dataset in (
        ("AND", and_gate),
        ("OR", or_gate),
    ):
        xs, ys = dataset()
        model = MLP(inputs=2, layers=[1])
        summary = train_binary_classifier_summary(model, xs, ys, steps=300, lr=0.2)

        print(f"{name} final loss: {summary.final_loss:.6f}")
        print(f"{name} accuracy:   {summary.accuracy:.3f}")
        print(f"{name} runtime:    {summary.elapsed_seconds:.4f}s")


def xor_demo() -> None:
    xs, ys = xor_gate()
    model = MLP(inputs=2, layers=[4, 1])
    summary = train_binary_classifier_summary(model, xs, ys, steps=1000, lr=0.2)
    probabilities = binary_probabilities(model, xs)

    print("\nXOR demo")
    print_summary(summary)

    for x, probability in zip(xs, probabilities):
        print(f"{x} -> {probability:.3f}")


def print_summary(summary) -> None:
    print(f"initial loss: {summary.initial_loss:.6f}")
    print(f"final loss:   {summary.final_loss:.6f}")
    if summary.accuracy is not None:
        print(f"accuracy:     {summary.accuracy:.3f}")
    print(f"runtime:      {summary.elapsed_seconds:.4f}s")


def main() -> None:
    random.seed(0)

    regression_demo()
    noisy_regression_demo()
    classification_demo()
    cluster_demo()
    logic_gate_demo()
    xor_demo()


if __name__ == "__main__":
    main()
