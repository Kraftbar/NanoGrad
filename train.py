"""Training helpers for the first scalar models."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from engine import Value
from metrics import binary_accuracy
from nn import Module
from optim import SGD


@dataclass(frozen=True)
class TrainingSummary:
    """Small training report for demos and smoke checks."""

    history: list[float]
    elapsed_seconds: float
    accuracy: float | None = None

    @property
    def initial_loss(self) -> float:
        return self.history[0]

    @property
    def final_loss(self) -> float:
        return self.history[-1]


def mse_loss(predictions: list[Value], targets: list[float]) -> Value:
    """Mean squared error written as explicit scalar equations."""

    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")

    losses = [
        (prediction - target) ** 2
        for prediction, target in zip(predictions, targets)
    ]
    return sum(losses, Value(0.0)) / len(losses)


def binary_cross_entropy(
    probabilities: list[Value],
    targets: list[float],
    *,
    eps: float = 1e-12,
) -> Value:
    """Binary cross entropy for scalar probabilities."""

    if len(probabilities) != len(targets):
        raise ValueError("probabilities and targets must have the same length")

    losses = [
        -(
            target * (probability + eps).log()
            + (1 - target) * (1 - probability + eps).log()
        )
        for probability, target in zip(probabilities, targets)
    ]
    return sum(losses, Value(0.0)) / len(losses)


def train_mse(
    model: Module,
    xs: list[list[float]],
    ys: list[float],
    *,
    steps: int = 100,
    lr: float = 0.05,
) -> list[float]:
    """Train a scalar-output model and return the loss history."""

    optimizer = SGD(model.parameters(), lr=lr)
    history = []

    for _ in range(steps):
        predictions = [model(x) for x in xs]
        if not all(isinstance(prediction, Value) for prediction in predictions):
            raise TypeError("train_mse expects the model to return one scalar Value")

        loss = mse_loss(predictions, ys)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(loss.data)

    return history


def train_mse_summary(
    model: Module,
    xs: list[list[float]],
    ys: list[float],
    *,
    steps: int = 100,
    lr: float = 0.05,
) -> TrainingSummary:
    """Train with MSE and return loss history plus elapsed runtime."""

    start = perf_counter()
    history = train_mse(model, xs, ys, steps=steps, lr=lr)
    elapsed_seconds = perf_counter() - start

    return TrainingSummary(
        history=history,
        elapsed_seconds=elapsed_seconds,
    )


def train_binary_classifier(
    model: Module,
    xs: list[list[float]],
    ys: list[float],
    *,
    steps: int = 100,
    lr: float = 0.05,
) -> list[float]:
    """Train a scalar-logit binary classifier and return BCE history."""

    optimizer = SGD(model.parameters(), lr=lr)
    history = []

    for _ in range(steps):
        logits = [model(x) for x in xs]
        if not all(isinstance(logit, Value) for logit in logits):
            raise TypeError(
                "train_binary_classifier expects the model to return one scalar Value"
            )

        probabilities = [logit.sigmoid() for logit in logits]
        loss = binary_cross_entropy(probabilities, ys)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(loss.data)

    return history


def train_binary_classifier_summary(
    model: Module,
    xs: list[list[float]],
    ys: list[float],
    *,
    steps: int = 100,
    lr: float = 0.05,
) -> TrainingSummary:
    """Train a binary classifier and return loss, accuracy, and runtime."""

    start = perf_counter()
    history = train_binary_classifier(model, xs, ys, steps=steps, lr=lr)
    elapsed_seconds = perf_counter() - start
    accuracy = binary_accuracy(binary_probabilities(model, xs), ys)

    return TrainingSummary(
        history=history,
        elapsed_seconds=elapsed_seconds,
        accuracy=accuracy,
    )


def binary_probabilities(model: Module, xs: list[list[float]]) -> list[float]:
    """Return sigmoid probabilities for a scalar-logit binary model."""

    probabilities = []
    for x in xs:
        logit = model(x)
        if not isinstance(logit, Value):
            raise TypeError("binary probability prediction expects one scalar Value")
        probabilities.append(logit.sigmoid().data)
    return probabilities
