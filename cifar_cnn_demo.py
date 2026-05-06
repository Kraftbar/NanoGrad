"""Train a tiny CIFAR-10 CNN from local binary batch files."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_cifar10_batches
from mnist_cnn_demo import SimpleCNN
from mnist_demo import print_confusion_matrix, print_epoch_report
from train import train_tensor_multiclass_dataset


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


def run(args: argparse.Namespace) -> None:
    train_paths = find_cifar10_files(args.data_dir)
    test_path = find_cifar10_test_file(args.data_dir, required=False)
    dataset = load_cifar10_batches(train_paths, limit=args.limit)
    validation_dataset = (
        None
        if test_path is None
        else load_cifar10_batches([test_path], limit=args.validation_limit)
    )

    if args.check_data:
        print_data_check(
            dataset,
            train_paths,
            validation_dataset=validation_dataset,
            test_path=test_path,
        )
        return

    model = SimpleCNN(
        image_shape=dataset.feature_shape,
        classes=CIFAR10_CLASSES,
        filters=args.filters,
        kernel_size=args.kernel_size,
        pool_size=args.pool_size,
        activation=args.activation,
        seed=args.seed,
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
    print(f"architecture: simple")
    print(f"activation:   {args.activation}")
    print(f"filters:      {args.filters}")
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
) -> None:
    """Print local CIFAR-10 file and shape information without training."""

    print("CIFAR-10 data check")
    print(f"train batches: {len(train_paths)}")
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
    parser.add_argument("--activation", choices=("relu", "tanh"), default="relu")
    parser.add_argument("--filters", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--pool-size", type=int, default=2)
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
