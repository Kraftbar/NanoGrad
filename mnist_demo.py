"""Train a tiny MNIST MLP from local IDX files."""

from __future__ import annotations

from pathlib import Path

from datasets import load_mnist
from tensor_nn import TensorMLP
from train import train_tensor_multiclass_dataset


DATA_DIR = Path("data/mnist")
TRAIN_IMAGES = DATA_DIR / "train-images-idx3-ubyte.gz"
TRAIN_LABELS = DATA_DIR / "train-labels-idx1-ubyte.gz"


def main() -> None:
    if not TRAIN_IMAGES.exists() or not TRAIN_LABELS.exists():
        raise SystemExit(
            "MNIST files not found. Expected local IDX gzip files at "
            f"{TRAIN_IMAGES} and {TRAIN_LABELS}."
        )

    dataset = load_mnist(TRAIN_IMAGES, TRAIN_LABELS, limit=512)
    model = TensorMLP(inputs=28 * 28, layers=[32, 10])
    summary = train_tensor_multiclass_dataset(
        model,
        dataset,
        epochs=2,
        batch_size=32,
        lr=0.05,
        shuffle=True,
        seed=0,
    )

    print("MNIST MLP demo")
    print(f"samples:      {len(dataset)}")
    print(f"initial loss: {summary.initial_loss:.6f}")
    print(f"final loss:   {summary.final_loss:.6f}")
    print(f"accuracy:     {summary.accuracy:.3f}")
    print(f"runtime:      {summary.elapsed_seconds:.4f}s")


if __name__ == "__main__":
    main()
