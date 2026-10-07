"""Recurrent networks: an LSTM and a character-level LSTM language model.

Before transformers, language models read text one token at a time and
carried a summary of everything so far in a hidden state (Karpathy's
"char-rnn" trained these on Tiny Shakespeare). The LSTM (Hochreiter and
Schmidhuber, 1997) is the classic recurrent cell. Each step it reads the
input x_t and the previous hidden state h, and computes four gates:

    i, f, g, o = split(x_t W_x^T + h W_h^T + b)       (each H wide)
    c = sigmoid(f) * c + sigmoid(i) * tanh(g)         the memory cell
    h = sigmoid(o) * tanh(c)                          the output

The forget gate f decides how much of the old memory to keep, the input
gate i how much new content g to write, and the output gate o how much of
the memory to show. Because c is updated by addition, gradients flow back
through many steps far better than in a plain RNN.

Training unrolls the loop over the sequence and backpropagates through
every step ("backpropagation through time"); autograd does that for free.
Gate order (i, f, g, o) matches PyTorch's ``nn.LSTM``.
"""

from __future__ import annotations

import math
import random

from tensor import Tensor, concat, matmul
from tensor_nn import TensorEmbedding, TensorLinear, TensorModule


class LSTM(TensorModule):
    """One LSTM layer over (batch, time, inputs) -> (batch, time, hidden)."""

    def __init__(self, inputs: int, hidden: int, *, seed: int = 0) -> None:
        self.inputs = inputs
        self.hidden = hidden
        # PyTorch's default init: uniform(-1/sqrt(H), 1/sqrt(H)) for everything.
        rng = random.Random(seed)
        bound = 1 / math.sqrt(hidden)

        def uniform(shape: tuple[int, ...]) -> Tensor:
            count = math.prod(shape)
            return Tensor([rng.uniform(-bound, bound) for _ in range(count)], shape, requires_grad=True)

        self.input_weight = uniform((4 * hidden, inputs))     # W_x: input -> gates
        self.hidden_weight = uniform((4 * hidden, hidden))    # W_h: previous h -> gates
        self.bias = uniform((4 * hidden,))

    def __call__(self, inputs: Tensor) -> Tensor:
        batch, length, _ = inputs.shape
        hidden_size = self.hidden
        # The input part of every gate, for all time steps in one product.
        input_gates = matmul(inputs, self.input_weight.T) + self.bias        # (B, T, 4H)
        steps = input_gates.split([1] * length, axis=1)

        hidden_weight_t = self.hidden_weight.T            # transpose once, not every step
        h = Tensor.zeros((batch, hidden_size))
        c = Tensor.zeros((batch, hidden_size))
        outputs = []
        for step in steps:
            gates = step.reshape((batch, 4 * hidden_size)) + matmul(h, hidden_weight_t)
            i, f, g, o = gates.split([hidden_size] * 4, axis=-1)
            c = f.sigmoid() * c + i.sigmoid() * g.tanh()
            h = o.sigmoid() * c.tanh()
            outputs.append(h.reshape((batch, 1, hidden_size)))
        return concat(outputs, axis=1)

    def parameters(self) -> list[Tensor]:
        return [self.input_weight, self.hidden_weight, self.bias]

    def state_dict(self) -> dict:
        return {
            "inputs": self.inputs,
            "hidden": self.hidden,
            "input_weight": list(self.input_weight.data),
            "hidden_weight": list(self.hidden_weight.data),
            "bias": list(self.bias.data),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("inputs") != self.inputs or state.get("hidden") != self.hidden:
            raise ValueError("state does not match LSTM sizes")
        for name in ("input_weight", "hidden_weight", "bias"):
            parameter = getattr(self, name)
            if len(state[name]) != parameter.numel:
                raise ValueError(f"state {name} does not match LSTM")
            parameter.data = [float(value) for value in state[name]]


class CharLSTMModel(TensorModule):
    """char-rnn: embedding -> stacked LSTM layers -> next-character scores."""

    def __init__(
        self,
        vocab_size: int,
        *,
        embedding_dim: int = 64,
        hidden: int = 256,
        num_layers: int = 2,
        seed: int = 0,
    ) -> None:
        self.vocab_size = vocab_size
        self.embedding = TensorEmbedding(vocab_size, embedding_dim, seed=seed)
        self.layers = [
            LSTM(embedding_dim if index == 0 else hidden, hidden, seed=seed + 1 + index)
            for index in range(num_layers)
        ]
        self.head = TensorLinear(hidden, vocab_size, seed=seed + 1 + num_layers)

    def sequence_logits(self, indices) -> Tensor:
        """Next-character scores at every position: (batch, time, vocab)."""

        hidden = self.embedding(indices)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.head(hidden)

    def __call__(self, indices) -> Tensor:
        """Scores after the last position: (batch, vocab)."""

        logits = self.sequence_logits(indices)
        batch, length, vocab = logits.shape
        return logits.split([length - 1, 1], axis=1)[1].reshape((batch, vocab)) if length > 1 else logits.reshape((batch, vocab))

    def parameters(self) -> list[Tensor]:
        return [
            *self.embedding.parameters(),
            *[parameter for layer in self.layers for parameter in layer.parameters()],
            *self.head.parameters(),
        ]

    def state_dict(self) -> dict:
        return {
            "embedding": self.embedding.state_dict(),
            "layers": [layer.state_dict() for layer in self.layers],
            "head": self.head.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.embedding.load_state_dict(state["embedding"])
        for layer, layer_state in zip(self.layers, state["layers"], strict=True):
            layer.load_state_dict(layer_state)
        self.head.load_state_dict(state["head"])
