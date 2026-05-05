"""Train a tiny MNIST CNN from local IDX files."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_mnist
from mnist_demo import DATA_DIR, find_mnist_files, print_epoch_report
from tensor import Tensor, avg_pool2d
from tensor_nn import TensorConv2D, TensorLinear, TensorModule
from train import train_tensor_multiclass_dataset


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
        features = self.conv(inputs).relu()
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


def run(args: argparse.Namespace) -> None:
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

    classes = int(max(dataset.ys)) + 1
    model = MNISTCNN(
        image_shape=dataset.feature_shape,
        classes=classes,
        filters=args.filters,
        kernel_size=args.kernel_size,
        pool_size=args.pool_size,
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

    print("MNIST CNN demo")
    print(f"images:       {images_path}")
    print(f"labels:       {labels_path}")
    print(f"samples:      {len(dataset)}")
    print(f"input shape:  {dataset.feature_shape}")
    print(f"classes:      {classes}")
    print(f"filters:      {args.filters}")
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
    parser.add_argument("--filters", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--pool-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-every", type=int, default=0)
    parser.add_argument("--save-model", type=Path)
    parser.add_argument("--load-model", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    try:
        run(parse_args(argv))
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
