"""Training helpers for the first scalar models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

from engine import Value
from losses import (
    binary_cross_entropy as tensor_binary_cross_entropy,
    softmax_cross_entropy,
)
from metrics import binary_accuracy, tensor_binary_accuracy, tensor_multiclass_accuracy
from nn import Module
from optim import SGD, TensorSGD
from tensor import Tensor
from datasets import TinyDataset


@dataclass(frozen=True)
class TrainingSummary:
    """Small training report for demos and smoke checks."""

    history: list[float]
    elapsed_seconds: float
    accuracy: float | None = None
    validation_accuracy: float | None = None

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


def train_tensor_binary_classifier(
    forward: Callable[[], Tensor],
    targets: Tensor,
    parameters: list[Tensor],
    *,
    steps: int = 100,
    lr: float = 0.05,
) -> TrainingSummary:
    """Train a tensor binary classifier from a closure over inputs and weights."""

    optimizer = TensorSGD(parameters, lr=lr)
    history = []
    start = perf_counter()

    for _ in range(steps):
        probabilities = forward()
        loss = tensor_binary_cross_entropy(probabilities, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(loss[0])

    elapsed_seconds = perf_counter() - start
    accuracy = tensor_binary_accuracy(forward(), targets)

    return TrainingSummary(
        history=history,
        elapsed_seconds=elapsed_seconds,
        accuracy=accuracy,
    )


def train_tensor_multiclass_classifier(
    forward: Callable[[], Tensor],
    targets: Tensor,
    parameters: list[Tensor],
    *,
    steps: int = 100,
    lr: float = 0.05,
) -> TrainingSummary:
    """Train a tensor multiclass classifier from a logits closure."""

    optimizer = TensorSGD(parameters, lr=lr)
    history = []
    start = perf_counter()

    for _ in range(steps):
        logits = forward()
        loss = softmax_cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(loss[0])

    elapsed_seconds = perf_counter() - start
    accuracy = tensor_multiclass_accuracy(forward(), targets)

    return TrainingSummary(
        history=history,
        elapsed_seconds=elapsed_seconds,
        accuracy=accuracy,
    )


def train_tensor_multiclass_dataset(
    model,
    dataset: TinyDataset,
    *,
    validation_dataset: TinyDataset | None = None,
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 0.05,
    shuffle: bool = True,
    seed: int | None = None,
) -> TrainingSummary:
    """Train a tensor multiclass model on TinyDataset batches."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")

    optimizer = TensorSGD(model.parameters(), lr=lr)
    history = []
    start = perf_counter()

    for epoch in range(epochs):
        batch_seed = None if seed is None else seed + epoch
        for xs, ys in dataset.batches(batch_size, shuffle=shuffle, seed=batch_seed):
            inputs = Tensor.from_list(xs)
            targets = Tensor.from_list(ys)
            logits = model(inputs)
            loss = softmax_cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            history.append(loss[0])

    elapsed_seconds = perf_counter() - start
    accuracy = _tensor_multiclass_dataset_accuracy(model, dataset)
    validation_accuracy = (
        None
        if validation_dataset is None
        else _tensor_multiclass_dataset_accuracy(model, validation_dataset)
    )

    return TrainingSummary(
        history=history,
        elapsed_seconds=elapsed_seconds,
        accuracy=accuracy,
        validation_accuracy=validation_accuracy,
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


def _tensor_multiclass_dataset_accuracy(model, dataset: TinyDataset) -> float:
    inputs = Tensor.from_list(dataset.xs)
    targets = Tensor.from_list(dataset.ys)
    return tensor_multiclass_accuracy(model(inputs), targets)
