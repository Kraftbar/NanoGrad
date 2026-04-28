"""Training helpers for the first scalar models."""

from __future__ import annotations

from engine import Value
from nn import Module
from optim import SGD


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
