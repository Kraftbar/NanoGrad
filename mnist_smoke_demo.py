"""Run a tiny MNIST-shaped smoke demo without external data files."""

from __future__ import annotations

from datasets import TinyDataset
from train import train_tensor_multiclass_dataset
from vision import SimpleCNN


def smoke_dataset() -> TinyDataset:
    """Return a tiny channel-first image dataset with two simple classes."""

    return TinyDataset(
        xs=[
            [
                [
                    [0.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0],
                ],
            ],
            [
                [
                    [1.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 1.0],
                ],
            ],
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
            ],
            [
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0],
                ],
            ],
        ],
        ys=[
            0,
            0,
            1,
            1,
        ],
    )


class SmokeCNN(SimpleCNN):
    """Tiny CNN used to exercise channel-first image batches."""

    def __init__(self, *, seed: int = 0) -> None:
        super().__init__(
            image_shape=(1, 4, 4),
            classes=2,
            filters=2,
            kernel_size=2,
            pool_size=3,
            seed=seed,
        )


def main() -> None:
    dataset = smoke_dataset()
    model = SmokeCNN(seed=0)
    summary = train_tensor_multiclass_dataset(
        model,
        dataset,
        epochs=250,
        batch_size=2,
        lr=0.15,
        shuffle=True,
        seed=0,
    )

    print("MNIST-style CNN smoke demo")
    print(f"samples:      {len(dataset)}")
    print(f"input shape:  {dataset.feature_shape}")
    print("classes:      2")
    print(f"initial loss: {summary.initial_loss:.6f}")
    print(f"final loss:   {summary.final_loss:.6f}")
    print(f"accuracy:     {summary.accuracy:.3f}")
    print(f"runtime:      {summary.elapsed_seconds:.4f}s")
    if summary.examples_per_second is not None:
        print(f"samples/s:    {summary.examples_per_second:.1f}")


if __name__ == "__main__":
    main()
