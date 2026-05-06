"""Small vision models built on top of the tensor core."""

from __future__ import annotations

from tensor import Tensor, avg_pool2d, max_pool2d
from tensor_nn import TensorConv2D, TensorLinear, TensorModule


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


class SimpleCNN(TensorModule):
    """Small channel-first CNN for local image experiments."""

    def __init__(
        self,
        *,
        image_shape: tuple[int, ...],
        classes: int,
        filters: int = 4,
        kernel_size: int = 3,
        pool_size: int = 2,
        pooling: str = "avg",
        activation: str = "relu",
        seed: int = 0,
    ) -> None:
        if len(image_shape) != 3:
            raise ValueError("SimpleCNN expects image_shape=(channels, rows, cols)")
        if classes <= 0:
            raise ValueError("classes must be positive")
        if filters <= 0:
            raise ValueError("filters must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if pooling not in ("avg", "max"):
            raise ValueError("pooling must be 'avg' or 'max'")
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
        self.pooling = pooling
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
        pooled = _pool(features, self.pool_size, pooling=self.pooling)
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


MNISTCNN = SimpleCNN


class TwoConvCNN(TensorModule):
    """Two-convolution channel-first CNN for small image experiments."""

    def __init__(
        self,
        *,
        image_shape: tuple[int, ...],
        classes: int,
        filters: int = 6,
        second_filters: int = 16,
        kernel_size: int = 5,
        pool_size: int = 2,
        pooling: str = "avg",
        activation: str = "tanh",
        seed: int = 0,
    ) -> None:
        if len(image_shape) != 3:
            raise ValueError("TwoConvCNN expects image_shape=(channels, rows, cols)")
        if classes <= 0:
            raise ValueError("classes must be positive")
        if filters <= 0 or second_filters <= 0:
            raise ValueError("filters must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if pooling not in ("avg", "max"):
            raise ValueError("pooling must be 'avg' or 'max'")
        if activation not in ("relu", "tanh"):
            raise ValueError("activation must be 'relu' or 'tanh'")

        channels, rows, cols = image_shape
        conv1_rows, conv1_cols = _valid_conv_shape(rows, cols, kernel_size)
        pool1_rows, pool1_cols = _pool_shape(conv1_rows, conv1_cols, pool_size)
        conv2_rows, conv2_cols = _valid_conv_shape(pool1_rows, pool1_cols, kernel_size)
        pool2_rows, pool2_cols = _pool_shape(conv2_rows, conv2_cols, pool_size)
        classifier_inputs = second_filters * pool2_rows * pool2_cols

        self.pool_size = pool_size
        self.pooling = pooling
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
        pooled = _pool(features, self.pool_size, pooling=self.pooling)
        features = _activate(self.conv2(pooled), self.activation)
        pooled = _pool(features, self.pool_size, pooling=self.pooling)
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


LeNetishCNN = TwoConvCNN


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


def _pool(tensor: Tensor, pool_size: int, *, pooling: str = "avg") -> Tensor:
    if pooling == "avg":
        return avg_pool2d(
            tensor,
            (pool_size, pool_size),
            stride=(pool_size, pool_size),
        )
    if pooling == "max":
        return max_pool2d(
            tensor,
            (pool_size, pool_size),
            stride=(pool_size, pool_size),
        )
    raise ValueError(f"unknown pooling: {pooling}")


def _activate(tensor: Tensor, activation: str) -> Tensor:
    if activation == "relu":
        return tensor.relu()
    if activation == "tanh":
        return tensor.tanh()
    raise ValueError(f"unknown activation: {activation}")
