"""A small MobileNet: depthwise separable convolutions, sized for CIFAR-10.

A 3x3 convolution from C to C' channels does 9 * C * C' multiply-adds per
pixel. MobileNet (Howard et al., 2017) splits it in two:

    depthwise 3x3: each channel filtered on its own      9 * C
    pointwise 1x1: channels mixed, every pixel alone     C * C'

about 8-9 times less work for nearly the same accuracy, which is why
phones and microcontrollers run it (MLPerf Tiny's visual wake words and
keyword spotting models are built this way). Each half is followed by
batch norm and ReLU.
"""

from __future__ import annotations

from tensor import Tensor
from tensor_nn import TensorBatchNorm2d, TensorConv2D, TensorDepthwiseConv2D, TensorLinear, TensorModule


class SeparableBlock(TensorModule):
    """relu(norm2(pointwise(relu(norm1(depthwise(x))))))"""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1, seed: int = 0) -> None:
        self.depthwise = TensorDepthwiseConv2D(in_channels, 3, stride=stride, padding=1, seed=seed)
        self.norm1 = TensorBatchNorm2d(in_channels)
        self.pointwise = TensorConv2D((out_channels, in_channels, 1, 1), use_bias=False, seed=seed + 1)
        self.norm2 = TensorBatchNorm2d(out_channels)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.norm1(self.depthwise(x)).relu()
        return self.norm2(self.pointwise(x)).relu()

    def layers(self) -> list[TensorModule]:
        return [self.depthwise, self.norm1, self.pointwise, self.norm2]

    def parameters(self) -> list[Tensor]:
        return [parameter for layer in self.layers() for parameter in layer.parameters()]


class MobileNet(TensorModule):
    """stem -> separable blocks (stride 2 twice) -> global average pool -> linear."""

    STAGES = ((32, 1), (64, 2), (64, 1), (128, 2), (128, 1))     # (output channels, stride)

    def __init__(self, *, image_shape: tuple[int, int, int] = (3, 32, 32), classes: int = 10,
                 stem: int = 16, seed: int = 0) -> None:
        self.stem = TensorConv2D((stem, image_shape[0], 3, 3), padding=1, use_bias=False, seed=seed)
        self.stem_norm = TensorBatchNorm2d(stem)
        self.blocks, channels = [], stem
        for index, (width, stride) in enumerate(self.STAGES):
            self.blocks.append(SeparableBlock(channels, width, stride=stride, seed=seed + 10 * (index + 1)))
            channels = width
        self.classifier = TensorLinear(channels, classes, seed=seed + 100)

    def __call__(self, images: Tensor) -> Tensor:
        x = self.stem_norm(self.stem(images)).relu()
        for block in self.blocks:
            x = block(x)
        batch, channels, rows, cols = x.shape
        return self.classifier(x.reshape((batch, channels, rows * cols)).mean(axis=-1))

    def parameters(self) -> list[Tensor]:
        return [*self.stem.parameters(), *self.stem_norm.parameters(),
                *[p for block in self.blocks for p in block.parameters()], *self.classifier.parameters()]
