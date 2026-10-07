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
from language import CausalSelfAttention, CharTransformerModel, TokenPositionEmbedding, TransformerBlock
from resnet import ResidualBlock, SmallResNet
from tensor_nn import TensorBatchNorm2d, TensorConv2D, TensorLayerNorm, TensorLinear, TensorModule, TensorRMSNorm
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
            (CharTransformerModel, self._char_transformer),
            (TensorRMSNorm, self._rms_norm),
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
        if layer.position_embedding is None:
            return tokens
        return tokens + self._p(layer.position_embedding.weight)[: indices.shape[-1]]

    def _rms_norm(self, layer: TensorRMSNorm, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + layer.eps) * self._p(layer.weight)

    def _norm(self, layer, x):
        return self._rms_norm(layer, x) if isinstance(layer, TensorRMSNorm) else self._layer_norm(layer, x)

    @staticmethod
    def _rope(x, base: float = 10000.0):
        """LLaMA's formulation: view each (even, odd) pair as a complex number
        and multiply by e^(i * position * base^(-2k/D))."""

        length, width = x.shape[-2:]
        frequencies = base ** (-torch.arange(0, width, 2, dtype=x.dtype) / width)
        angles = torch.outer(torch.arange(length, dtype=x.dtype), frequencies)
        pairs = torch.view_as_complex(x.reshape(*x.shape[:-1], width // 2, 2).contiguous())
        return torch.view_as_real(pairs * torch.polar(torch.ones_like(angles), angles)).reshape(x.shape)

    def _layer_norm(self, layer: TensorLayerNorm, x):
        return F.layer_norm(x, (layer.normalized_shape,), self._p(layer.weight), self._p(layer.bias), layer.eps)

    def _attention(self, layer: CausalSelfAttention, x):
        context = x.shape[1]
        heads = []
        for head, query in enumerate(layer.queries):
            shared = head // layer.group_size        # grouped-query attention shares keys and values
            key, value = layer.keys[shared], layer.values[shared]
            q, k, v = self._linear(query, x), self._linear(key, x), self._linear(value, x)
            if layer.rotary:
                q, k = self._rope(q), self._rope(k)
            scores = q @ k.transpose(1, 2) * (1 / math.sqrt(layer.head_dim))
            if layer.causal:
                future = torch.ones(context, context, dtype=torch.bool).triu(1)
                scores = scores + torch.zeros(context, context, dtype=x.dtype).masked_fill(future, -1e9)
            heads.append(scores.softmax(dim=-1) @ v)
        return self._linear(layer.projection, torch.cat(heads, dim=-1))

    def _transformer_block(self, block: TransformerBlock, x):
        x = x + self._attention(block.attention, self._norm(block.norm1, x))
        normed = self._norm(block.norm2, x)
        hidden = self._linear(block.feed_forward_in, normed)
        if block.feed_forward_activation == "swiglu":
            hidden = F.silu(hidden) * self._linear(block.feed_forward_up, normed)
        else:
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

    def _char_transformer(self, model: CharTransformerModel, indices):
        """All positions' next-token logits, like ``sequence_logits``."""

        hidden = self._token_position(model.embedding, indices)
        for block in model.blocks:
            hidden = self._transformer_block(block, hidden)
        return self._linear(model.projection, self._norm(model.norm, hidden))

    # -- diffusion U-Net (diffusion.py); takes images and time steps --------

    def unet(self, model, images, steps: list[int]):
        from diffusion import timestep_embedding

        silu = F.silu
        embedding = torch.tensor(list(timestep_embedding(steps, model.time_width).data), dtype=self.dtype)
        time = self._linear(model.time_out, silu(self._linear(model.time_in, embedding.reshape(len(steps), -1))))

        def block(b, x):
            hidden = silu(self._conv(b.conv1, x))
            hidden = hidden + self._linear(b.time, time).reshape(x.shape[0], b.out_channels, 1, 1)
            hidden = silu(self._conv(b.conv2, hidden))
            return hidden + (x if b.skip is None else self._conv(b.skip, x))

        up = lambda x: F.interpolate(x, scale_factor=2, mode="nearest")
        skip1 = block(model.down1, self._conv(model.stem, images))
        skip2 = block(model.down2, self._conv(model.shrink1, skip1))
        hidden = block(model.middle, self._conv(model.shrink2, skip2))
        hidden = block(model.up2, torch.cat([up(hidden), skip2], dim=1))
        hidden = block(model.up1, torch.cat([up(hidden), skip1], dim=1))
        return self._conv(model.head, hidden)
