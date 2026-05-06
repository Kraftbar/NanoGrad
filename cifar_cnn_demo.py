"""Train a tiny CIFAR-10 CNN from local binary batch files."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import (
    channel_first_stats,
    load_cifar10_batches,
    normalize_channel_first,
)
from mnist_demo import print_confusion_matrix, print_epoch_report
from train import train_tensor_multiclass_dataset
from vision import SimpleCNN, TwoConvCNN


DATA_DIR = Path("data/cifar10")
TRAIN_BATCH_NAMES = (
    "data_batch_1.bin",
    "data_batch_2.bin",
    "data_batch_3.bin",
    "data_batch_4.bin",
    "data_batch_5.bin",
)
TEST_BATCH_NAME = "test_batch.bin"
CIFAR10_CLASSES = 10
EXTRACTED_DIR_NAME = "cifar-10-batches-bin"
CIFAR_PRESETS = {
    "tiny": {
        "normalize": "none",
        "architecture": "simple",
        "activation": "relu",
        "filters": 4,
        "second_filters": 8,
        "kernel_size": 3,
        "pool_size": 2,
        "pooling": "avg",
    },
    "tiny-normalized": {
        "normalize": "train",
        "architecture": "simple",
        "activation": "relu",
        "filters": 4,
        "second_filters": 8,
        "kernel_size": 3,
        "pool_size": 2,
        "pooling": "avg",
    },
    "two-conv-normalized": {
        "normalize": "train",
        "architecture": "two-conv",
        "activation": "relu",
        "filters": 4,
        "second_filters": 8,
        "kernel_size": 3,
        "pool_size": 2,
        "pooling": "avg",
    },
    "two-conv-maxpool-normalized": {
        "normalize": "train",
        "architecture": "two-conv",
        "activation": "relu",
        "filters": 4,
        "second_filters": 8,
        "kernel_size": 3,
        "pool_size": 2,
        "pooling": "max",
    },
}


def run(args: argparse.Namespace) -> None:
    args = _apply_preset(args)
    train_paths = find_cifar10_files(args.data_dir)
    test_path = find_cifar10_test_file(args.data_dir, required=False)
    dataset = load_cifar10_batches(train_paths, limit=args.limit)
    validation_dataset = (
        None
        if test_path is None
        else load_cifar10_batches([test_path], limit=args.validation_limit)
    )
    normalization_stats = None
    if args.normalize == "train":
        normalization_stats = channel_first_stats(dataset)
        dataset = normalize_channel_first(dataset, *normalization_stats)
        if validation_dataset is not None:
            validation_dataset = normalize_channel_first(
                validation_dataset,
                *normalization_stats,
            )

    if args.check_data:
        print_data_check(
            dataset,
            train_paths,
            validation_dataset=validation_dataset,
            test_path=test_path,
            normalization=args.normalize,
            normalization_stats=normalization_stats,
        )
        return

    model = build_model(
        args,
        image_shape=dataset.feature_shape,
        classes=CIFAR10_CLASSES,
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

    print("CIFAR-10 CNN demo")
    print(f"batches:      {len(train_paths)}")
    print(f"samples:      {len(dataset)}")
    print(f"input shape:  {dataset.feature_shape}")
    print(f"classes:      {CIFAR10_CLASSES}")
    print(f"preset:       {args.preset}")
    print(f"normalize:    {args.normalize}")
    if normalization_stats is not None:
        print_channel_stats(normalization_stats)
    print(f"architecture: {args.architecture}")
    print(f"activation:   {args.activation}")
    print(f"pooling:      {args.pooling}")
    print(f"filters:      {args.filters}")
    if args.architecture == "two-conv":
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


def print_data_check(
    dataset,
    train_paths: list[Path],
    *,
    validation_dataset=None,
    test_path: Path | None = None,
    normalization: str = "none",
    normalization_stats: tuple[list[float], list[float]] | None = None,
) -> None:
    """Print local CIFAR-10 file and shape information without training."""

    print("CIFAR-10 data check")
    print(f"train batches: {len(train_paths)}")
    print(f"normalize:     {normalization}")
    if normalization_stats is not None:
        print_channel_stats(normalization_stats)
    print_dataset_summary("train", dataset)

    if validation_dataset is None or test_path is None:
        print("test:          not found")
        return

    print(f"test batch:    {test_path}")
    print_dataset_summary("test", validation_dataset)


def print_dataset_summary(name: str, dataset) -> None:
    labels = sorted({int(label) for label in dataset.ys})
    print(f"{name} samples: {len(dataset)}")
    print(f"{name} shape:   {dataset.feature_shape}")
    print(f"{name} labels:  {labels}")


def print_channel_stats(stats: tuple[list[float], list[float]]) -> None:
    means, stds = stats
    mean_text = ", ".join(f"{value:.4f}" for value in means)
    std_text = ", ".join(f"{value:.4f}" for value in stds)
    print(f"channel mean: [{mean_text}]")
    print(f"channel std:  [{std_text}]")


def build_model(
    args: argparse.Namespace,
    *,
    image_shape: tuple[int, ...],
    classes: int,
):
    if args.architecture == "simple":
        return SimpleCNN(
            image_shape=image_shape,
            classes=classes,
            filters=args.filters,
            kernel_size=args.kernel_size,
            pool_size=args.pool_size,
            pooling=args.pooling,
            activation=args.activation,
            seed=args.seed,
        )
    if args.architecture == "two-conv":
        return TwoConvCNN(
            image_shape=image_shape,
            classes=classes,
            filters=args.filters,
            second_filters=args.second_filters,
            kernel_size=args.kernel_size,
            pool_size=args.pool_size,
            pooling=args.pooling,
            activation=args.activation,
            seed=args.seed,
        )
    raise ValueError(f"unknown architecture: {args.architecture}")


def _apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset = CIFAR_PRESETS[args.preset]
    if args.normalize is None:
        args.normalize = preset["normalize"]
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
    if args.pooling is None:
        args.pooling = preset["pooling"]
    return args


def find_cifar10_files(data_dir: Path, *, required: bool = True) -> list[Path]:
    paths = []
    for candidate_dir in _cifar10_data_dirs(data_dir):
        paths = [
            candidate_dir / name
            for name in TRAIN_BATCH_NAMES
            if (candidate_dir / name).exists()
        ]
        if paths:
            break
    if not paths and required:
        raise FileNotFoundError(
            f"CIFAR-10 batch files not found under {data_dir}."
        )
    return paths


def find_cifar10_test_file(
    data_dir: Path,
    *,
    required: bool = True,
) -> Path | None:
    for candidate_dir in _cifar10_data_dirs(data_dir):
        path = candidate_dir / TEST_BATCH_NAME
        if path.exists():
            return path
    if required:
        raise FileNotFoundError(f"CIFAR-10 test batch not found under {data_dir}.")
    return None


def _cifar10_data_dirs(data_dir: Path) -> list[Path]:
    return [
        data_dir,
        data_dir / EXTRACTED_DIR_NAME,
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--validation-limit", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--preset", choices=sorted(CIFAR_PRESETS), default="tiny")
    parser.add_argument(
        "--normalize",
        choices=("none", "train"),
        help="Normalize channel-first images with stats from the training split.",
    )
    parser.add_argument(
        "--architecture",
        choices=("simple", "two-conv"),
    )
    parser.add_argument("--activation", choices=("relu", "tanh"))
    parser.add_argument("--filters", type=int)
    parser.add_argument("--second-filters", type=int)
    parser.add_argument("--kernel-size", type=int)
    parser.add_argument("--pool-size", type=int)
    parser.add_argument("--pooling", choices=("avg", "max"))
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
        help="Verify local CIFAR-10 files and channel-first shapes without training.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    try:
        run(parse_args(argv))
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
