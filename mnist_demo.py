"""Train a tiny MNIST MLP from local IDX files."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_mnist
from tensor_nn import TensorMLP
from train import train_tensor_multiclass_dataset


DATA_DIR = Path("data/mnist")
TRAIN_IMAGE_NAMES = (
    "train-images-idx3-ubyte.gz",
    "train-images-idx3-ubyte",
)
TRAIN_LABEL_NAMES = (
    "train-labels-idx1-ubyte.gz",
    "train-labels-idx1-ubyte",
)
TEST_IMAGE_NAMES = (
    "t10k-images-idx3-ubyte.gz",
    "t10k-images-idx3-ubyte",
)
TEST_LABEL_NAMES = (
    "t10k-labels-idx1-ubyte.gz",
    "t10k-labels-idx1-ubyte",
)


def run(args: argparse.Namespace) -> None:
    images_path, labels_path = find_mnist_files(args.data_dir)
    dataset = load_mnist(images_path, labels_path, limit=args.limit)
    validation_paths = find_mnist_files(args.data_dir, split="test", required=False)
    validation_dataset = (
        None
        if validation_paths is None
        else load_mnist(
            validation_paths[0],
            validation_paths[1],
            limit=args.validation_limit,
        )
    )

    inputs = len(dataset.xs[0])
    classes = int(max(dataset.ys)) + 1
    model = TensorMLP(inputs=inputs, layers=[args.hidden, classes])
    summary = train_tensor_multiclass_dataset(
        model,
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        shuffle=True,
        seed=args.seed,
        validation_dataset=validation_dataset,
    )

    print("MNIST MLP demo")
    print(f"images:       {images_path}")
    print(f"labels:       {labels_path}")
    print(f"samples:      {len(dataset)}")
    print(f"inputs:       {inputs}")
    print(f"classes:      {classes}")
    print(f"initial loss: {summary.initial_loss:.6f}")
    print(f"final loss:   {summary.final_loss:.6f}")
    print(f"accuracy:     {summary.accuracy:.3f}")
    if summary.validation_accuracy is not None:
        print(f"val accuracy: {summary.validation_accuracy:.3f}")
    print(f"runtime:      {summary.elapsed_seconds:.4f}s")


def find_mnist_files(
    data_dir: Path,
    *,
    split: str = "train",
    required: bool = True,
) -> tuple[Path, Path] | None:
    if split == "train":
        image_names = TRAIN_IMAGE_NAMES
        label_names = TRAIN_LABEL_NAMES
    elif split == "test":
        image_names = TEST_IMAGE_NAMES
        label_names = TEST_LABEL_NAMES
    else:
        raise ValueError("MNIST split must be 'train' or 'test'")

    images_path = _first_existing(data_dir, image_names)
    labels_path = _first_existing(data_dir, label_names)

    if images_path is None or labels_path is None:
        if not required:
            return None
        raise FileNotFoundError(
            f"MNIST files not found. Expected {split} image and label IDX files "
            f"under {data_dir}."
        )

    return images_path, labels_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--validation-limit", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    try:
        run(parse_args(argv))
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error


def _first_existing(data_dir: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = data_dir / name
        if path.exists():
            return path
    return None


if __name__ == "__main__":
    main()
