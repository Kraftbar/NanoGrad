"""A tiny BERT-style encoder, trained by masked character modeling.

BERT (Devlin et al., 2018) is the model family behind the GLUE and SQuAD
benchmark results. It is GPT's transformer with two changes:

- attention looks both ways: every token sees the whole sequence, not just
  the tokens before it (``causal=False`` in ``language.py``'s blocks)
- the training task is *masked* language modeling. Hide about 15% of the
  tokens and predict the originals from the context on both sides. Of the
  chosen positions, 80% become a special [MASK] token, 10% a random token,
  and 10% stay as they are, so the model cannot rely on seeing [MASK].

Here the tokens are characters (as in the character GPT), and [MASK] is one
extra token id after the alphabet.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from language import TokenPositionEmbedding, TransformerBlock
from losses import softmax_cross_entropy
from tensor import Tensor, gather_rows
from tensor_nn import TensorLayerNorm, TensorLinear, TensorModule


class TinyBERT(TensorModule):
    """embedding -> bidirectional transformer blocks -> norm -> per-token character scores."""

    def __init__(
        self,
        vocab_size: int,
        *,
        context_size: int = 64,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        seed: int = 0,
    ) -> None:
        self.vocab_size = vocab_size
        self.mask_id = vocab_size                      # one extra input token
        self.context_size = context_size
        self.embedding = TokenPositionEmbedding(vocab_size + 1, context_size, embedding_dim, seed=seed)
        self.blocks = [
            TransformerBlock(
                embedding_dim,
                context_size,
                num_heads=num_heads,
                feed_forward_activation="gelu",
                causal=False,
                seed=seed + 2 + 8 * layer,
            )
            for layer in range(num_layers)
        ]
        self.norm = TensorLayerNorm(embedding_dim)
        self.head = TensorLinear(embedding_dim, vocab_size, seed=seed + 99)

    def __call__(self, indices: Sequence[Sequence[int]]) -> Tensor:
        """Character scores for every position: (batch, context, vocab)."""

        hidden = self.embedding(indices)
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(self.norm(hidden))

    def parameters(self) -> list[Tensor]:
        return [
            *self.embedding.parameters(),
            *[parameter for block in self.blocks for parameter in block.parameters()],
            *self.norm.parameters(),
            *self.head.parameters(),
        ]

    def state_dict(self) -> dict:
        return {
            "embedding": self.embedding.state_dict(),
            "blocks": [block.state_dict() for block in self.blocks],
            "norm": self.norm.state_dict(),
            "head": self.head.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.embedding.load_state_dict(state["embedding"])
        for block, block_state in zip(self.blocks, state["blocks"], strict=True):
            block.load_state_dict(block_state)
        self.norm.load_state_dict(state["norm"])
        self.head.load_state_dict(state["head"])


def mask_tokens(
    sequences: Sequence[Sequence[int]],
    *,
    vocab_size: int,
    mask_id: int,
    rng: random.Random,
    mask_rate: float = 0.15,
) -> tuple[list[list[int]], list[int], list[int]]:
    """BERT's masking recipe.

    Returns (inputs, positions, targets): the corrupted sequences, the flat
    positions (row * length + column) the loss is computed at, and the
    original characters there. Every sequence gets at least one position.
    """

    inputs: list[list[int]] = []
    positions: list[int] = []
    targets: list[int] = []
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        chosen = [column for column in range(length) if rng.random() < mask_rate]
        if not chosen:
            chosen = [rng.randrange(length)]
        corrupted = list(sequence)
        for column in chosen:
            roll = rng.random()
            if roll < 0.8:
                corrupted[column] = mask_id
            elif roll < 0.9:
                corrupted[column] = rng.randrange(vocab_size)
            # else: keep the original character
            positions.append(row * length + column)
            targets.append(sequence[column])
        inputs.append(corrupted)
    return inputs, positions, targets


def masked_loss(model: TinyBERT, inputs, positions: list[int], targets: list[int]) -> Tensor:
    """Mean cross entropy at the masked positions only."""

    logits = model(inputs)
    batch, length, vocab = logits.shape
    chosen = gather_rows(logits.reshape((batch * length, vocab)), positions)
    return softmax_cross_entropy(chosen, targets)
