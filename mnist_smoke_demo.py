"""Run a tiny MNIST-shaped smoke demo without external data files."""

from __future__ import annotations

from datasets import TinyDataset
from tensor import Tensor, avg_pool2d
from tensor_nn import TensorConv2D, TensorLinear
from train import train_tensor_multiclass_dataset


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


class SmokeCNN:
    """Tiny CNN used to exercise channel-first image batches."""

    def __init__(self, *, seed: int = 0) -> None:
        self.conv = TensorConv2D((2, 1, 2, 2), seed=seed)
        self.classifier = TensorLinear(inputs=2, outputs=2, seed=seed + 1)

    def __call__(self, inputs: Tensor) -> Tensor:
        features = self.conv(inputs).relu()
        pooled = avg_pool2d(features, (3, 3))
        return self.classifier(pooled.flatten(start_axis=1))

    def parameters(self) -> list[Tensor]:
        return [
            *self.conv.parameters(),
            *self.classifier.parameters(),
        ]


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


if __name__ == "__main__":
    main()
