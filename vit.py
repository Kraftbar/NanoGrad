"""A tiny vision transformer (ViT), sized for MNIST.

ViT (Dosovitskiy et al., 2020) reads an image the way GPT reads text: cut
the image into square patches, turn each patch into a token vector with one
linear layer, add a learned position vector per patch, and run transformer
blocks. Two differences from the character GPT in ``language.py``, which
provides the blocks:

- attention is not causal: every patch may look at every other patch
- the output is one label per image, so the tokens are averaged (mean
  pooling) and a linear layer maps the average to class scores

A 28x28 MNIST digit with 7x7 patches becomes 16 tokens of 49 pixels.
"""

from __future__ import annotations

import random

from language import TransformerBlock
from tensor import Tensor
from tensor_nn import TensorLayerNorm, TensorLinear, TensorModule


class TinyViT(TensorModule):
    """patches -> linear embedding + positions -> transformer blocks -> mean -> linear."""

    def __init__(
        self,
        *,
        image_shape: tuple[int, int, int] = (1, 28, 28),
        patch_size: int = 7,
        classes: int = 10,
        embedding_dim: int = 32,
        num_heads: int = 2,
        num_layers: int = 2,
        seed: int = 0,
    ) -> None:
        channels, rows, cols = image_shape
        if rows % patch_size or cols % patch_size:
            raise ValueError("image sides must be multiples of patch_size")
        self.image_shape = image_shape
        self.patch_size = patch_size
        self.num_patches = (rows // patch_size) * (cols // patch_size)
        patch_dim = channels * patch_size * patch_size

        self.patch_embedding = TensorLinear(patch_dim, embedding_dim, seed=seed)
        rng = random.Random(seed + 1)
        self.position = Tensor(
            [rng.gauss(0.0, 0.02) for _ in range(self.num_patches * embedding_dim)],
            (self.num_patches, embedding_dim),
            requires_grad=True,
        )
        self.blocks = [
            TransformerBlock(
                embedding_dim,
                self.num_patches,
                num_heads=num_heads,
                feed_forward_activation="gelu",
                causal=False,
                seed=seed + 10 * (layer + 1),
            )
            for layer in range(num_layers)
        ]
        self.norm = TensorLayerNorm(embedding_dim)
        self.classifier = TensorLinear(embedding_dim, classes, seed=seed + 99)

    def patchify(self, images: Tensor) -> Tensor:
        """(B, C, H, W) -> (B, patches, C * p * p), patches in row-major order."""

        batch = images.shape[0]
        channels, rows, cols = self.image_shape
        p = self.patch_size
        grid = images.reshape((batch, channels, rows // p, p, cols // p, p))
        # Bring the two patch-grid axes forward, then flatten each patch.
        patches = grid.permute((0, 2, 4, 1, 3, 5))
        return patches.reshape((batch, self.num_patches, channels * p * p))

    def __call__(self, images: Tensor) -> Tensor:
        tokens = self.patch_embedding(self.patchify(images)) + self.position
        for block in self.blocks:
            tokens = block(tokens)
        pooled = self.norm(tokens).mean(axis=1)              # (B, embedding)
        return self.classifier(pooled)

    def parameters(self) -> list[Tensor]:
        return [
            *self.patch_embedding.parameters(),
            self.position,
            *[parameter for block in self.blocks for parameter in block.parameters()],
            *self.norm.parameters(),
            *self.classifier.parameters(),
        ]

    def state_dict(self) -> dict:
        return {
            "patch_embedding": self.patch_embedding.state_dict(),
            "position": list(self.position.data),
            "blocks": [block.state_dict() for block in self.blocks],
            "norm": self.norm.state_dict(),
            "classifier": self.classifier.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.patch_embedding.load_state_dict(state["patch_embedding"])
        if len(state["position"]) != self.position.numel:
            raise ValueError("state position does not match TinyViT")
        self.position.data = [float(value) for value in state["position"]]
        for block, block_state in zip(self.blocks, state["blocks"], strict=True):
            block.load_state_dict(block_state)
        self.norm.load_state_dict(state["norm"])
        self.classifier.load_state_dict(state["classifier"])
