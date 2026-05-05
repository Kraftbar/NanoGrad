"""Run a tiny MNIST-shaped smoke demo without external data files."""

from __future__ import annotations

from datasets import TinyDataset
from tensor_nn import TensorMLP
from train import train_tensor_multiclass_dataset


def smoke_dataset() -> TinyDataset:
    """Return a tiny 2x2 image dataset with two simple classes."""

    return TinyDataset(
        xs=[
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.1, 0.9],
        ],
        ys=[
            0,
            0,
            1,
            1,
        ],
    )


def main() -> None:
    dataset = smoke_dataset()
    model = TensorMLP(inputs=4, layers=[4, 2], seed=0)
    summary = train_tensor_multiclass_dataset(
        model,
        dataset,
        epochs=200,
        batch_size=2,
        lr=0.2,
        shuffle=True,
        seed=0,
    )

    print("MNIST-style smoke demo")
    print(f"samples:      {len(dataset)}")
    print("inputs:       4")
    print("classes:      2")
    print(f"initial loss: {summary.initial_loss:.6f}")
    print(f"final loss:   {summary.final_loss:.6f}")
    print(f"accuracy:     {summary.accuracy:.3f}")
    print(f"runtime:      {summary.elapsed_seconds:.4f}s")


if __name__ == "__main__":
    main()
