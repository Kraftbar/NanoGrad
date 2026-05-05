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
from metrics import (
    binary_accuracy,
    tensor_binary_accuracy,
    tensor_multiclass_accuracy,
    tensor_multiclass_confusion_matrix,
)
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
    evaluation_loss: float | None = None
    validation_loss: float | None = None
    examples_seen: int | None = None
    confusion_matrix: list[list[int]] | None = None

    @property
    def initial_loss(self) -> float:
        return self.history[0]

    @property
    def final_loss(self) -> float:
        return self.history[-1]

    @property
    def examples_per_second(self) -> float | None:
        if self.examples_seen is None:
            return None
        if self.elapsed_seconds <= 0.0:
            return 0.0
        return self.examples_seen / self.elapsed_seconds


@dataclass(frozen=True)
class EvaluationSummary:
    """Whole-dataset classifier metrics."""

    loss: float
    accuracy: float
    elapsed_seconds: float
    examples_seen: int
    confusion_matrix: list[list[int]] | None = None

    @property
    def examples_per_second(self) -> float:
        if self.elapsed_seconds <= 0.0:
            return 0.0
        return self.examples_seen / self.elapsed_seconds


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
    epoch_callback: Callable[[int, TrainingSummary], None] | None = None,
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

        if epoch_callback is not None:
            epoch_elapsed = perf_counter() - start
            train_eval = evaluate_tensor_multiclass_dataset(
                model,
                dataset,
                batch_size=batch_size,
            )
            validation_eval = (
                None
                if validation_dataset is None
                else evaluate_tensor_multiclass_dataset(
                    model,
                    validation_dataset,
                    batch_size=batch_size,
                )
            )
            epoch_summary = TrainingSummary(
                history=history[:],
                elapsed_seconds=epoch_elapsed,
                accuracy=train_eval.accuracy,
                validation_accuracy=(
                    None
                    if validation_eval is None
                    else validation_eval.accuracy
                ),
                evaluation_loss=train_eval.loss,
                validation_loss=(
                    None
                    if validation_eval is None
                    else validation_eval.loss
                ),
                examples_seen=len(dataset) * (epoch + 1),
                confusion_matrix=train_eval.confusion_matrix,
            )
            epoch_callback(epoch + 1, epoch_summary)

    elapsed_seconds = perf_counter() - start
    train_eval = evaluate_tensor_multiclass_dataset(
        model,
        dataset,
        batch_size=batch_size,
    )
    validation_eval = (
        None
        if validation_dataset is None
        else evaluate_tensor_multiclass_dataset(
            model,
            validation_dataset,
            batch_size=batch_size,
        )
    )

    return TrainingSummary(
        history=history,
        elapsed_seconds=elapsed_seconds,
        accuracy=train_eval.accuracy,
        validation_accuracy=(
            None
            if validation_eval is None
            else validation_eval.accuracy
        ),
        evaluation_loss=train_eval.loss,
        validation_loss=(
            None
            if validation_eval is None
            else validation_eval.loss
        ),
        examples_seen=len(dataset) * epochs,
        confusion_matrix=train_eval.confusion_matrix,
    )


def evaluate_tensor_multiclass_dataset(
    model,
    dataset: TinyDataset,
    *,
    batch_size: int = 128,
) -> EvaluationSummary:
    """Evaluate multiclass loss and accuracy across a full TinyDataset."""

    start = perf_counter()
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0
    confusion_matrix: list[list[int]] | None = None

    for xs, ys in dataset.batches(batch_size, shuffle=False):
        inputs = Tensor.from_list(xs)
        targets = Tensor.from_list(ys)
        logits = model(inputs)
        loss = softmax_cross_entropy(logits, targets)
        batch_count = len(ys)

        total_loss += loss[0] * batch_count
        total_correct += tensor_multiclass_accuracy(logits, targets) * batch_count
        confusion_matrix = _add_confusion_matrices(
            confusion_matrix,
            tensor_multiclass_confusion_matrix(logits, targets),
        )
        total_count += batch_count

    return EvaluationSummary(
        loss=total_loss / total_count,
        accuracy=total_correct / total_count,
        elapsed_seconds=perf_counter() - start,
        examples_seen=total_count,
        confusion_matrix=confusion_matrix,
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


def _add_confusion_matrices(
    left: list[list[int]] | None,
    right: list[list[int]],
) -> list[list[int]]:
    if left is None:
        return [
            row[:]
            for row in right
        ]
    if len(left) != len(right) or any(
        len(left_row) != len(right_row)
        for left_row, right_row in zip(left, right)
    ):
        raise ValueError("confusion matrix shapes must match")
    return [
        [
            left_value + right_value
            for left_value, right_value in zip(left_row, right_row)
        ]
        for left_row, right_row in zip(left, right)
    ]
