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
from tensor_nn import TensorModule
from train import train_tensor_multiclass_dataset
from vision import CNN_PRESETS, LeNetishCNN, SimpleCNN


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
        print_confusion_matrix(summary.confusion_matrix, label="train confusion")
    if args.show_confusion and summary.validation_confusion_matrix is not None:
        print_confusion_matrix(
            summary.validation_confusion_matrix,
            label="val confusion",
        )

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
        return SimpleCNN(
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


def main(argv: list[str] | None = None) -> None:
    try:
        run(parse_args(argv))
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
