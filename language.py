"""Reusable small language-model modules."""

from __future__ import annotations

from tensor import Tensor
from tensor_nn import TensorEmbedding, TensorLinear, TensorModule


class CharBigramModel(TensorModule):
    """Character model: flattened one-hot context -> next-char logits."""

    def __init__(
        self,
        vocab_size: int,
        *,
        context_size: int = 1,
        seed: int = 0,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if context_size <= 0:
            raise ValueError("context_size must be positive")
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.projection = TensorLinear(
            vocab_size * context_size,
            vocab_size,
            seed=seed,
        )

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.projection(inputs)

    def parameters(self) -> list[Tensor]:
        return self.projection.parameters()

    def state_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "context_size": self.context_size,
            "projection": self.projection.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("vocab_size") != self.vocab_size:
            raise ValueError("state vocab_size does not match CharBigramModel")
        if state.get("context_size") != self.context_size:
            raise ValueError("state context_size does not match CharBigramModel")
        self.projection.load_state_dict(state["projection"])


class CharEmbeddingModel(TensorModule):
    """Character model: token-id context -> embeddings -> next-char logits."""

    def __init__(
        self,
        vocab_size: int,
        *,
        context_size: int = 1,
        embedding_dim: int = 16,
        seed: int = 0,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if context_size <= 0:
            raise ValueError("context_size must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.embedding = TensorEmbedding(vocab_size, embedding_dim, seed=seed)
        self.projection = TensorLinear(
            embedding_dim * context_size,
            vocab_size,
            seed=seed + 1,
        )

    def __call__(self, inputs: Tensor) -> Tensor:
        embedded = self.embedding(inputs)
        return self.projection(embedded.flatten(start_axis=1))

    def parameters(self) -> list[Tensor]:
        return [
            *self.embedding.parameters(),
            *self.projection.parameters(),
        ]

    def state_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "context_size": self.context_size,
            "embedding_dim": self.embedding_dim,
            "embedding": self.embedding.state_dict(),
            "projection": self.projection.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("vocab_size") != self.vocab_size:
            raise ValueError("state vocab_size does not match CharEmbeddingModel")
        if state.get("context_size") != self.context_size:
            raise ValueError("state context_size does not match CharEmbeddingModel")
        if state.get("embedding_dim") != self.embedding_dim:
            raise ValueError("state embedding_dim does not match CharEmbeddingModel")
        self.embedding.load_state_dict(state["embedding"])
        self.projection.load_state_dict(state["projection"])
