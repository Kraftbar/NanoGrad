"""LoRA: fine-tune a large model by training a small low-rank correction.

Fine-tuning normally updates every weight. LoRA (Hu et al., 2021) freezes
the pretrained weight W (out x in) and learns only a low-rank update:

    W' = W + (alpha / r) * B A        A: (r x in), B: (out x r), r << in, out

B starts at zero, so training starts exactly from the pretrained model,
and A starts small and random. Only A and B train: for a 64 x 64 layer at
rank 4 that is 512 numbers instead of 4,096. After training, B A can be
added into W once ("merged"), so the fine-tuned model is as fast as the
original.

``add_lora`` puts adapters on the query and value projections of every
attention layer in a model, which is the setup of the LoRA paper.
"""

from __future__ import annotations

import math
import random

from tensor import Tensor, matmul
from tensor_nn import TensorLinear, TensorModule, linear


class LoRALinear(TensorModule):
    """A frozen ``TensorLinear`` plus a trainable rank-``rank`` update."""

    def __init__(self, base: TensorLinear, *, rank: int = 4, alpha: float | None = None, seed: int = 0) -> None:
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        outputs, inputs = base.weight.shape
        self.base = base
        self.rank = rank
        self.scale = (rank if alpha is None else alpha) / rank
        base.weight.requires_grad = False                    # the pretrained layer stays fixed
        base.bias.requires_grad = False
        rng = random.Random(seed)
        bound = 1 / math.sqrt(inputs)                         # A like a fresh linear layer, B zero
        self.down = Tensor([rng.uniform(-bound, bound) for _ in range(rank * inputs)], (rank, inputs), requires_grad=True)
        self.up = Tensor.zeros((outputs, rank), requires_grad=True)

    @property
    def weight(self) -> Tensor:
        """W + (alpha / r) B A: what the layer computes with (and what attention concatenates)."""

        return self.base.weight + matmul(self.up, self.down) * self.scale

    @property
    def bias(self) -> Tensor:
        return self.base.bias

    def __call__(self, inputs: Tensor) -> Tensor:
        return linear(inputs, self.weight, self.bias)

    def parameters(self) -> list[Tensor]:
        return [self.down, self.up]

    def merged(self) -> TensorLinear:
        """A plain linear layer with the update folded into its weight."""

        outputs, inputs = self.base.weight.shape
        weight = Tensor(list(self.weight.data), (outputs, inputs), requires_grad=True)
        return TensorLinear(inputs, outputs, weight=weight, bias=Tensor(list(self.bias.data), (outputs,), requires_grad=True))


def add_lora(model: TensorModule, *, rank: int = 4, alpha: float | None = None, seed: int = 0) -> list[Tensor]:
    """Freeze ``model`` and put LoRA adapters on every attention layer's query
    and value projections. Returns the adapters' parameters (the only ones
    left to train)."""

    from language import CausalSelfAttention

    for parameter in model.parameters():
        parameter.requires_grad = False
    adapters = []
    for module in model.modules():
        if isinstance(module, CausalSelfAttention):
            for layers in (module.queries, module.values):
                for index, layer in enumerate(layers):
                    layers[index] = LoRALinear(layer, rank=rank, alpha=alpha, seed=seed + len(adapters))
                    adapters.append(layers[index])
    return [parameter for adapter in adapters for parameter in adapter.parameters()]


def merge_lora(model: TensorModule) -> None:
    """Fold every adapter into its layer (inference afterwards costs nothing extra)."""

    from language import CausalSelfAttention

    for module in model.modules():
        if isinstance(module, CausalSelfAttention):
            for layers in (module.queries, module.values):
                for index, layer in enumerate(layers):
                    if isinstance(layer, LoRALinear):
                        layers[index] = layer.merged()
