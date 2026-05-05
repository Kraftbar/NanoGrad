"""Train a tiny MNIST CNN from local IDX files."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_mnist
from mnist_demo import (
    DATA_DIR,
    find_mnist_files,
    print_confusion_matrix,
    print_epoch_report,
)
from tensor import Tensor, avg_pool2d
from tensor_nn import TensorConv2D, TensorLinear, TensorModule
from train import train_tensor_multiclass_dataset


CNN_PRESETS = {
    "tiny": {
        "activation": "relu",
        "architecture": "simple",
        "filters": 4,
        "kernel_size": 3,
        "pool_size": 2,
        "second_filters": None,
    },
    "lenet-ish": {
        "activation": "tanh",
        "architecture": "lenet-ish",
        "filters": 6,
        "kernel_size": 5,
        "pool_size": 2,
        "second_filters": 16,
    },
}


class MNISTCNN(TensorModule):
    """Small channel-first CNN for local MNIST experiments."""

    def __init__(
        self,
        *,
        image_shape: tuple[int, ...],
        classes: int,
        filters: int = 4,
        kernel_size: int = 3,
        pool_size: int = 2,
        activation: str = "relu",
        seed: int = 0,
    ) -> None:
        if len(image_shape) != 3:
            raise ValueError("MNISTCNN expects image_shape=(channels, rows, cols)")
        if classes <= 0:
            raise ValueError("classes must be positive")
        if filters <= 0:
            raise ValueError("filters must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if activation not in ("relu", "tanh"):
            raise ValueError("activation must be 'relu' or 'tanh'")

        channels, rows, cols = image_shape
        conv_rows = rows - kernel_size + 1
        conv_cols = cols - kernel_size + 1
        if conv_rows <= 0 or conv_cols <= 0:
            raise ValueError("kernel_size must fit inside image_shape")
        if pool_size > conv_rows or pool_size > conv_cols:
            raise ValueError("pool_size must fit inside convolution output")

        pooled_rows = ((conv_rows - pool_size) // pool_size) + 1
        pooled_cols = ((conv_cols - pool_size) // pool_size) + 1
        classifier_inputs = filters * pooled_rows * pooled_cols

        self.pool_size = pool_size
        self.activation = activation
        self.conv = TensorConv2D(
            (filters, channels, kernel_size, kernel_size),
            seed=seed,
        )
        self.classifier = TensorLinear(
            inputs=classifier_inputs,
            outputs=classes,
            seed=seed + 1,
        )

    def __call__(self, inputs: Tensor) -> Tensor:
        features = _activate(self.conv(inputs), self.activation)
        pooled = avg_pool2d(
            features,
            (self.pool_size, self.pool_size),
            stride=(self.pool_size, self.pool_size),
        )
        return self.classifier(pooled.flatten(start_axis=1))

    def parameters(self) -> list[Tensor]:
        return [
            *self.conv.parameters(),
            *self.classifier.parameters(),
        ]

    def state_dict(self) -> dict:
        return {
            "conv": self.conv.state_dict(),
            "classifier": self.classifier.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.conv.load_state_dict(state["conv"])
        self.classifier.load_state_dict(state["classifier"])


class LeNetishCNN(TensorModule):
    """Two-convolution MNIST CNN shaped after the early LeNet pattern."""

    def __init__(
        self,
        *,
        image_shape: tuple[int, ...],
        classes: int,
        filters: int = 6,
        second_filters: int = 16,
        kernel_size: int = 5,
        pool_size: int = 2,
        activation: str = "tanh",
        seed: int = 0,
    ) -> None:
        if len(image_shape) != 3:
            raise ValueError("LeNetishCNN expects image_shape=(channels, rows, cols)")
        if classes <= 0:
            raise ValueError("classes must be positive")
        if filters <= 0 or second_filters <= 0:
            raise ValueError("filters must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if activation not in ("relu", "tanh"):
            raise ValueError("activation must be 'relu' or 'tanh'")

        channels, rows, cols = image_shape
        conv1_rows, conv1_cols = _valid_conv_shape(rows, cols, kernel_size)
        pool1_rows, pool1_cols = _pool_shape(conv1_rows, conv1_cols, pool_size)
        conv2_rows, conv2_cols = _valid_conv_shape(pool1_rows, pool1_cols, kernel_size)
        pool2_rows, pool2_cols = _pool_shape(conv2_rows, conv2_cols, pool_size)
        classifier_inputs = second_filters * pool2_rows * pool2_cols

        self.pool_size = pool_size
        self.activation = activation
        self.conv1 = TensorConv2D(
            (filters, channels, kernel_size, kernel_size),
            seed=seed,
        )
        self.conv2 = TensorConv2D(
            (second_filters, filters, kernel_size, kernel_size),
            seed=seed + 1,
        )
        self.classifier = TensorLinear(
            inputs=classifier_inputs,
            outputs=classes,
            seed=seed + 2,
        )

    def __call__(self, inputs: Tensor) -> Tensor:
        features = _activate(self.conv1(inputs), self.activation)
        pooled = _pool(features, self.pool_size)
        features = _activate(self.conv2(pooled), self.activation)
        pooled = _pool(features, self.pool_size)
        return self.classifier(pooled.flatten(start_axis=1))

    def parameters(self) -> list[Tensor]:
        return [
            *self.conv1.parameters(),
            *self.conv2.parameters(),
            *self.classifier.parameters(),
        ]

    def state_dict(self) -> dict:
        return {
            "conv1": self.conv1.state_dict(),
            "conv2": self.conv2.state_dict(),
            "classifier": self.classifier.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.conv1.load_state_dict(state["conv1"])
        self.conv2.load_state_dict(state["conv2"])
        self.classifier.load_state_dict(state["classifier"])


def run(args: argparse.Namespace) -> None:
    args = _apply_preset(args)
    images_path, labels_path = find_mnist_files(args.data_dir)
    dataset = load_mnist(
        images_path,
        labels_path,
        limit=args.limit,
        channel_first=True,
    )
    validation_paths = find_mnist_files(args.data_dir, split="test", required=False)
    validation_dataset = (
        None
        if validation_paths is None
        else load_mnist(
            validation_paths[0],
            validation_paths[1],
            limit=args.validation_limit,
            channel_first=True,
        )
    )

    if args.check_data:
        print_data_check(
            dataset,
            images_path,
            labels_path,
            validation_dataset=validation_dataset,
            validation_paths=validation_paths,
        )
        return

    classes = int(max(dataset.ys)) + 1
    model = build_model(
        args,
        image_shape=dataset.feature_shape,
        classes=classes,
    )
    if args.load_model is not None:
        model.load(args.load_model)

    summary = train_tensor_multiclass_dataset(
        model,
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        shuffle=True,
        seed=args.seed,
        validation_dataset=validation_dataset,
        epoch_callback=(
            None
            if args.report_every <= 0
            else lambda epoch, summary: print_epoch_report(
                epoch,
                args.epochs,
                summary,
                report_every=args.report_every,
            )
        ),
    )

    print("MNIST CNN demo")
    print(f"images:       {images_path}")
    print(f"labels:       {labels_path}")
    print(f"samples:      {len(dataset)}")
    print(f"input shape:  {dataset.feature_shape}")
    print(f"classes:      {classes}")
    print(f"preset:       {args.preset}")
    print(f"architecture: {args.architecture}")
    print(f"activation:   {args.activation}")
    print(f"filters:      {args.filters}")
    if args.second_filters is not None:
        print(f"filters 2:    {args.second_filters}")
    print(f"parameters:   {model.num_parameters()}")
    print(f"initial loss: {summary.initial_loss:.6f}")
    print(f"final batch:  {summary.final_loss:.6f}")
    if summary.evaluation_loss is not None:
        print(f"train loss:   {summary.evaluation_loss:.6f}")
    print(f"accuracy:     {summary.accuracy:.3f}")
    if summary.validation_loss is not None:
        print(f"val loss:     {summary.validation_loss:.6f}")
    if summary.validation_accuracy is not None:
        print(f"val accuracy: {summary.validation_accuracy:.3f}")
    print(f"runtime:      {summary.elapsed_seconds:.4f}s")
    if summary.examples_per_second is not None:
        print(f"samples/s:    {summary.examples_per_second:.1f}")
    if args.show_confusion and summary.confusion_matrix is not None:
        print_confusion_matrix(summary.confusion_matrix)

    if args.save_model is not None:
        model.save(args.save_model)
        print(f"saved model:  {args.save_model}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--validation-limit", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--preset", choices=sorted(CNN_PRESETS), default="tiny")
    parser.add_argument("--architecture", choices=("simple", "lenet-ish"))
    parser.add_argument("--activation", choices=("relu", "tanh"))
    parser.add_argument("--filters", type=int)
    parser.add_argument("--second-filters", type=int)
    parser.add_argument("--kernel-size", type=int)
    parser.add_argument("--pool-size", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-every", type=int, default=0)
    parser.add_argument("--save-model", type=Path)
    parser.add_argument("--load-model", type=Path)
    parser.add_argument(
        "--show-confusion",
        action="store_true",
        help="Print the train-set confusion matrix after evaluation.",
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Verify local MNIST files and channel-first shapes without training.",
    )
    return parser.parse_args(argv)


def print_data_check(
    dataset,
    images_path: Path,
    labels_path: Path,
    *,
    validation_dataset=None,
    validation_paths: tuple[Path, Path] | None = None,
) -> None:
    """Print local MNIST file and channel-first shape information."""

    print("MNIST CNN data check")
    print(f"train images: {images_path}")
    print(f"train labels: {labels_path}")
    print_dataset_summary("train", dataset)

    if validation_dataset is None or validation_paths is None:
        print("test:         not found")
        return

    print(f"test images:  {validation_paths[0]}")
    print(f"test labels:  {validation_paths[1]}")
    print_dataset_summary("test", validation_dataset)


def print_dataset_summary(name: str, dataset) -> None:
    labels = sorted({int(label) for label in dataset.ys})
    print(f"{name} samples: {len(dataset)}")
    print(f"{name} shape:   {dataset.feature_shape}")
    print(f"{name} labels:  {labels}")


def _apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset = CNN_PRESETS[args.preset]
    if args.architecture is None:
        args.architecture = preset["architecture"]
    if args.activation is None:
        args.activation = preset["activation"]
    if args.filters is None:
        args.filters = preset["filters"]
    if args.second_filters is None:
        args.second_filters = preset["second_filters"]
    if args.kernel_size is None:
        args.kernel_size = preset["kernel_size"]
    if args.pool_size is None:
        args.pool_size = preset["pool_size"]
    return args


def build_model(
    args: argparse.Namespace,
    *,
    image_shape: tuple[int, ...],
    classes: int,
) -> TensorModule:
    if args.architecture == "simple":
        return MNISTCNN(
            image_shape=image_shape,
            classes=classes,
            filters=args.filters,
            kernel_size=args.kernel_size,
            pool_size=args.pool_size,
            activation=args.activation,
            seed=args.seed,
        )
    if args.architecture == "lenet-ish":
        return LeNetishCNN(
            image_shape=image_shape,
            classes=classes,
            filters=args.filters,
            second_filters=args.second_filters,
            kernel_size=args.kernel_size,
            pool_size=args.pool_size,
            activation=args.activation,
            seed=args.seed,
        )
    raise ValueError(f"unknown architecture: {args.architecture}")


def _valid_conv_shape(rows: int, cols: int, kernel_size: int) -> tuple[int, int]:
    out_rows = rows - kernel_size + 1
    out_cols = cols - kernel_size + 1
    if out_rows <= 0 or out_cols <= 0:
        raise ValueError("kernel_size must fit inside image shape")
    return out_rows, out_cols


def _pool_shape(rows: int, cols: int, pool_size: int) -> tuple[int, int]:
    if pool_size > rows or pool_size > cols:
        raise ValueError("pool_size must fit inside feature map")
    return ((rows - pool_size) // pool_size) + 1, ((cols - pool_size) // pool_size) + 1


def _pool(tensor: Tensor, pool_size: int) -> Tensor:
    return avg_pool2d(
        tensor,
        (pool_size, pool_size),
        stride=(pool_size, pool_size),
    )


def _activate(tensor: Tensor, activation: str) -> Tensor:
    if activation == "relu":
        return tensor.relu()
    if activation == "tanh":
        return tensor.tanh()
    raise ValueError(f"unknown activation: {activation}")


def main(argv: list[str] | None = None) -> None:
    try:
        run(parse_args(argv))
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
