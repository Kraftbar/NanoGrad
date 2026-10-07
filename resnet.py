"""A small residual network (ResNet), sized for MNIST.

ResNet (He et al., 2015) is the network behind the classic ImageNet
benchmark (ResNet-50). Its idea: each block learns a *correction* that is
added to its input, rather than a whole new representation,

    out = relu(F(x) + shortcut(x))

so very deep stacks still train: the gradient always has the shortcut's
direct path back. F is two 3x3 convolutions, each followed by batch norm.
When a block halves the image (stride 2) or changes the channel count, the
shortcut becomes a 1x1 strided convolution so the shapes still match.

This version keeps the ResNet-18 recipe at a tiny scale: a 3x3 stem, three
stages of one block each (widths 8, 16, 32 by default; 28x28 -> 28 -> 14 -> 7
pixels), global average pooling, and one linear layer.
"""

from __future__ import annotations

from tensor import Tensor
from tensor_nn import TensorBatchNorm2d, TensorConv2D, TensorLinear, TensorModule


class ResidualBlock(TensorModule):
    """out = relu(norm2(conv2(relu(norm1(conv1(x))))) + shortcut(x))"""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1, seed: int = 0) -> None:
        self.conv1 = TensorConv2D(
            (out_channels, in_channels, 3, 3), stride=stride, padding=1, use_bias=False, seed=seed,
        )
        self.norm1 = TensorBatchNorm2d(out_channels)
        self.conv2 = TensorConv2D(
            (out_channels, out_channels, 3, 3), padding=1, use_bias=False, seed=seed + 1,
        )
        self.norm2 = TensorBatchNorm2d(out_channels)
        self.shortcut_conv = None
        self.shortcut_norm = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut_conv = TensorConv2D(
                (out_channels, in_channels, 1, 1), stride=stride, use_bias=False, seed=seed + 2,
            )
            self.shortcut_norm = TensorBatchNorm2d(out_channels)

    def __call__(self, inputs: Tensor) -> Tensor:
        hidden = self.norm1(self.conv1(inputs)).relu()
        correction = self.norm2(self.conv2(hidden))
        shortcut = inputs
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_norm(self.shortcut_conv(inputs))
        return (correction + shortcut).relu()

    def layers(self) -> list[TensorModule]:
        layers = [self.conv1, self.norm1, self.conv2, self.norm2]
        if self.shortcut_conv is not None:
            layers += [self.shortcut_conv, self.shortcut_norm]
        return layers

    def parameters(self) -> list[Tensor]:
        return [parameter for layer in self.layers() for parameter in layer.parameters()]

    def state_dict(self) -> dict:
        return {"layers": [layer.state_dict() for layer in self.layers()]}

    def load_state_dict(self, state: dict) -> None:
        for layer, layer_state in zip(self.layers(), state["layers"], strict=True):
            layer.load_state_dict(layer_state)


class SmallResNet(TensorModule):
    """stem -> three residual stages -> global average pool -> linear."""

    def __init__(
        self,
        *,
        image_shape: tuple[int, int, int] = (1, 28, 28),
        classes: int = 10,
        widths: tuple[int, int, int] = (8, 16, 32),
        seed: int = 0,
    ) -> None:
        channels = image_shape[0]
        first, second, third = widths
        self.stem = TensorConv2D((first, channels, 3, 3), padding=1, use_bias=False, seed=seed)
        self.stem_norm = TensorBatchNorm2d(first)
        self.blocks = [
            ResidualBlock(first, first, stride=1, seed=seed + 10),
            ResidualBlock(first, second, stride=2, seed=seed + 20),
            ResidualBlock(second, third, stride=2, seed=seed + 30),
        ]
        self.classifier = TensorLinear(third, classes, seed=seed + 40)

    def __call__(self, images: Tensor) -> Tensor:
        features = self.stem_norm(self.stem(images)).relu()
        for block in self.blocks:
            features = block(features)
        batch, channels, rows, cols = features.shape
        # Global average pooling: one number per channel, whatever the image size.
        pooled = features.reshape((batch, channels, rows * cols)).mean(axis=-1)
        return self.classifier(pooled)

    def parameters(self) -> list[Tensor]:
        return [
            *self.stem.parameters(),
            *self.stem_norm.parameters(),
            *[parameter for block in self.blocks for parameter in block.parameters()],
            *self.classifier.parameters(),
        ]

    def state_dict(self) -> dict:
        return {
            "stem": self.stem.state_dict(),
            "stem_norm": self.stem_norm.state_dict(),
            "blocks": [block.state_dict() for block in self.blocks],
            "classifier": self.classifier.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.stem.load_state_dict(state["stem"])
        self.stem_norm.load_state_dict(state["stem_norm"])
        for block, block_state in zip(self.blocks, state["blocks"], strict=True):
            block.load_state_dict(block_state)
        self.classifier.load_state_dict(state["classifier"])
