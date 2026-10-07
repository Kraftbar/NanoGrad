"""Run a NanoGrad model's architecture in PyTorch, with the same weights.

``TorchTwin(model)`` copies every NanoGrad parameter into a PyTorch tensor
and then computes the forward pass by walking the NanoGrad model's own
layer objects, calling the matching PyTorch function for each one. The
architecture is read from the model itself, so the twin cannot drift from
it, and in float64 the two should agree to rounding error. The comparison
scripts use this to check NanoGrad against PyTorch, and to time the same
work in both.

Supported: the vision models in ``vision.py`` (TwoConvCNN), ``resnet.py``
and ``vit.py``, the encoder in ``bert.py``, and the layers they are built
from. Requires PyTorch.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from bert import TinyBERT
from language import CausalSelfAttention, TokenPositionEmbedding, TransformerBlock
from resnet import ResidualBlock, SmallResNet
from tensor_nn import TensorBatchNorm2d, TensorConv2D, TensorLayerNorm, TensorLinear, TensorModule
from vision import TwoConvCNN
from vit import TinyViT


class TorchTwin:
    def __init__(self, model: TensorModule, *, dtype: torch.dtype = torch.float64) -> None:
        self.model = model
        self.dtype = dtype
        self.training = True
        self._tensors = {
            id(parameter): torch.tensor(list(parameter.data), dtype=dtype).reshape(parameter.shape).requires_grad_()
            for parameter in model.parameters()
        }
        # Same order as model.parameters(), so weights can be compared one to one.
        self.parameters = [self._tensors[id(parameter)] for parameter in model.parameters()]
        self._running = {
            id(module): (torch.tensor(module.running_mean, dtype=dtype), torch.tensor(module.running_var, dtype=dtype))
            for module in model.modules()
            if isinstance(module, TensorBatchNorm2d)
        }

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward(self.model, inputs)

    def running_statistics(self) -> list[list[float]]:
        """Batch-norm running means and variances, in module order."""

        return [values.tolist() for pair in self._running.values() for values in pair]

    def _p(self, parameter) -> torch.Tensor:
        return self._tensors[id(parameter)]

    # -- dispatch --------------------------------------------------------

    def forward(self, module: TensorModule, x: torch.Tensor) -> torch.Tensor:
        for kind, method in (
            (TwoConvCNN, self._two_conv_cnn),
            (SmallResNet, self._small_resnet),
            (ResidualBlock, self._residual_block),
            (TinyViT, self._tiny_vit),
            (TinyBERT, self._tiny_bert),
            (TokenPositionEmbedding, self._token_position),
            (TransformerBlock, self._transformer_block),
            (CausalSelfAttention, self._attention),
            (TensorConv2D, self._conv),
            (TensorBatchNorm2d, self._batch_norm),
            (TensorLayerNorm, self._layer_norm),
            (TensorLinear, self._linear),
        ):
            if isinstance(module, kind):
                return method(module, x)
        raise TypeError(f"TorchTwin does not know {type(module).__name__}")

    # -- layers ----------------------------------------------------------

    def _linear(self, layer: TensorLinear, x):
        return x @ self._p(layer.weight).T + self._p(layer.bias)

    def _conv(self, layer: TensorConv2D, x):
        out = F.conv2d(x, self._p(layer.kernel), stride=layer.stride, padding=layer.padding)
        return out if layer.bias is None else out + self._p(layer.bias)

    def _batch_norm(self, layer: TensorBatchNorm2d, x):
        running_mean, running_var = self._running[id(layer)]
        return F.batch_norm(
            x, running_mean, running_var, self._p(layer.weight), self._p(layer.bias),
            training=self.training, momentum=layer.momentum, eps=layer.eps,
        )

    def _token_position(self, layer: TokenPositionEmbedding, indices):
        tokens = self._p(layer.token_embedding.weight)[indices]
        return tokens + self._p(layer.position_embedding.weight)[: indices.shape[-1]]

    def _layer_norm(self, layer: TensorLayerNorm, x):
        return F.layer_norm(x, (layer.normalized_shape,), self._p(layer.weight), self._p(layer.bias), layer.eps)

    def _attention(self, layer: CausalSelfAttention, x):
        context = x.shape[1]
        heads = []
        for query, key, value in zip(layer.queries, layer.keys, layer.values):
            q, k, v = self._linear(query, x), self._linear(key, x), self._linear(value, x)
            scores = q @ k.transpose(1, 2) * (1 / math.sqrt(layer.head_dim))
            if layer.causal:
                future = torch.ones(context, context, dtype=torch.bool).triu(1)
                scores = scores + torch.zeros(context, context, dtype=x.dtype).masked_fill(future, -1e9)
            heads.append(scores.softmax(dim=-1) @ v)
        return self._linear(layer.projection, torch.cat(heads, dim=-1))

    def _transformer_block(self, block: TransformerBlock, x):
        x = x + self._attention(block.attention, self._layer_norm(block.norm1, x))
        hidden = self._linear(block.feed_forward_in, self._layer_norm(block.norm2, x))
        hidden = F.gelu(hidden) if block.feed_forward_activation == "gelu" else F.relu(hidden)
        return x + self._linear(block.feed_forward_out, hidden)

    # -- models ----------------------------------------------------------

    def _two_conv_cnn(self, model: TwoConvCNN, x):
        activate = torch.tanh if model.activation == "tanh" else F.relu
        pool = F.avg_pool2d if model.pooling == "avg" else F.max_pool2d
        x = pool(activate(self._conv(model.conv1, x)), model.pool_size)
        x = pool(activate(self._conv(model.conv2, x)), model.pool_size)
        return self._linear(model.classifier, x.flatten(1))

    def _residual_block(self, block: ResidualBlock, x):
        hidden = F.relu(self._batch_norm(block.norm1, self._conv(block.conv1, x)))
        correction = self._batch_norm(block.norm2, self._conv(block.conv2, hidden))
        shortcut = x
        if block.shortcut_conv is not None:
            shortcut = self._batch_norm(block.shortcut_norm, self._conv(block.shortcut_conv, x))
        return F.relu(correction + shortcut)

    def _small_resnet(self, model: SmallResNet, x):
        x = F.relu(self._batch_norm(model.stem_norm, self._conv(model.stem, x)))
        for block in model.blocks:
            x = self._residual_block(block, x)
        return self._linear(model.classifier, x.mean(dim=(2, 3)))

    def _tiny_vit(self, model: TinyViT, x):
        batch = x.shape[0]
        channels, rows, cols = model.image_shape
        p = model.patch_size
        patches = (
            x.reshape(batch, channels, rows // p, p, cols // p, p)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch, model.num_patches, channels * p * p)
        )
        tokens = self._linear(model.patch_embedding, patches) + self._p(model.position)
        for block in model.blocks:
            tokens = self._transformer_block(block, tokens)
        return self._linear(model.classifier, self._layer_norm(model.norm, tokens).mean(dim=1))

    def _tiny_bert(self, model: TinyBERT, indices):
        hidden = self._token_position(model.embedding, indices)
        for block in model.blocks:
            hidden = self._transformer_block(block, hidden)
        return self._linear(model.head, self._layer_norm(model.norm, hidden))
