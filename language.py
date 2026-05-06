"""Reusable small language-model modules."""

from __future__ import annotations

import math

from tensor import Tensor, matmul
from tensor_nn import TensorEmbedding, TensorLayerNorm, TensorLinear, TensorModule


class TokenPositionEmbedding(TensorModule):
    """Token embeddings plus learned positions for fixed-length contexts."""

    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        embedding_dim: int,
        *,
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
        self.token_embedding = TensorEmbedding(vocab_size, embedding_dim, seed=seed)
        self.position_embedding = TensorEmbedding(
            context_size,
            embedding_dim,
            seed=seed + 1,
        )

    def __call__(self, indices) -> Tensor:
        token_vectors = self.token_embedding(indices)
        if token_vectors.shape[-2] != self.context_size:
            raise ValueError("input last dimension must match context_size")

        position_vectors = self.position_embedding(
            list(range(self.context_size)),
        )
        return token_vectors + position_vectors

    def parameters(self) -> list[Tensor]:
        return [
            *self.token_embedding.parameters(),
            *self.position_embedding.parameters(),
        ]

    def state_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "context_size": self.context_size,
            "embedding_dim": self.embedding_dim,
            "token_embedding": self.token_embedding.state_dict(),
            "position_embedding": self.position_embedding.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("vocab_size") != self.vocab_size:
            raise ValueError("state vocab_size does not match TokenPositionEmbedding")
        if state.get("context_size") != self.context_size:
            raise ValueError("state context_size does not match TokenPositionEmbedding")
        if state.get("embedding_dim") != self.embedding_dim:
            raise ValueError("state embedding_dim does not match TokenPositionEmbedding")
        self.token_embedding.load_state_dict(state["token_embedding"])
        self.position_embedding.load_state_dict(state["position_embedding"])


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
        self.embedding = TokenPositionEmbedding(
            vocab_size,
            context_size,
            embedding_dim,
            seed=seed,
        )
        self.projection = TensorLinear(
            embedding_dim * context_size,
            vocab_size,
            seed=seed + 2,
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


class CausalSelfAttention(TensorModule):
    """Single-head causal self-attention for tiny sequence models."""

    def __init__(
        self,
        embedding_dim: int,
        context_size: int,
        *,
        seed: int = 0,
    ) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if context_size <= 0:
            raise ValueError("context_size must be positive")

        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.query = TensorLinear(embedding_dim, embedding_dim, seed=seed)
        self.key = TensorLinear(embedding_dim, embedding_dim, seed=seed + 1)
        self.value = TensorLinear(embedding_dim, embedding_dim, seed=seed + 2)
        self.projection = TensorLinear(embedding_dim, embedding_dim, seed=seed + 3)

    def __call__(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3:
            raise ValueError("CausalSelfAttention expects (batch, context, embedding)")

        batch_size, context_size, embedding_dim = inputs.shape
        if context_size != self.context_size:
            raise ValueError("input context dimension must match context_size")
        if embedding_dim != self.embedding_dim:
            raise ValueError("input embedding dimension must match embedding_dim")

        flat_inputs = inputs.reshape((batch_size * context_size, embedding_dim))
        query = self.query(flat_inputs)
        key = self.key(flat_inputs)
        value = self.value(flat_inputs)

        scores = matmul(query, key.T) * (1 / math.sqrt(embedding_dim))
        masked_scores = scores + _causal_attention_mask(batch_size, context_size)
        weights = masked_scores.softmax(axis=1)
        attended = matmul(weights, value)
        output = self.projection(attended)
        return output.reshape((batch_size, context_size, embedding_dim))

    def parameters(self) -> list[Tensor]:
        return [
            *self.query.parameters(),
            *self.key.parameters(),
            *self.value.parameters(),
            *self.projection.parameters(),
        ]

    def state_dict(self) -> dict:
        return {
            "embedding_dim": self.embedding_dim,
            "context_size": self.context_size,
            "query": self.query.state_dict(),
            "key": self.key.state_dict(),
            "value": self.value.state_dict(),
            "projection": self.projection.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("embedding_dim") != self.embedding_dim:
            raise ValueError("state embedding_dim does not match CausalSelfAttention")
        if state.get("context_size") != self.context_size:
            raise ValueError("state context_size does not match CausalSelfAttention")
        self.query.load_state_dict(state["query"])
        self.key.load_state_dict(state["key"])
        self.value.load_state_dict(state["value"])
        self.projection.load_state_dict(state["projection"])


def _causal_attention_mask(batch_size: int, context_size: int) -> Tensor:
    total_tokens = batch_size * context_size
    data = []
    for query_index in range(total_tokens):
        query_batch = query_index // context_size
        query_position = query_index % context_size
        for key_index in range(total_tokens):
            key_batch = key_index // context_size
            key_position = key_index % context_size
            can_attend = (
                query_batch == key_batch
                and key_position <= query_position
            )
            data.append(0.0 if can_attend else -1e9)
    return Tensor(data, (total_tokens, total_tokens))


class TransformerBlock(TensorModule):
    """Pre-norm transformer block for tiny causal language models."""

    def __init__(
        self,
        embedding_dim: int,
        context_size: int,
        *,
        hidden_dim: int | None = None,
        seed: int = 0,
    ) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if context_size <= 0:
            raise ValueError("context_size must be positive")
        if hidden_dim is None:
            hidden_dim = embedding_dim * 4
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.hidden_dim = hidden_dim
        self.norm1 = TensorLayerNorm(embedding_dim)
        self.attention = CausalSelfAttention(
            embedding_dim,
            context_size,
            seed=seed,
        )
        self.norm2 = TensorLayerNorm(embedding_dim)
        self.feed_forward_in = TensorLinear(
            embedding_dim,
            hidden_dim,
            seed=seed + 4,
        )
        self.feed_forward_out = TensorLinear(
            hidden_dim,
            embedding_dim,
            seed=seed + 5,
        )

    def __call__(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3:
            raise ValueError("TransformerBlock expects (batch, context, embedding)")

        batch_size, context_size, embedding_dim = inputs.shape
        if context_size != self.context_size:
            raise ValueError("input context dimension must match context_size")
        if embedding_dim != self.embedding_dim:
            raise ValueError("input embedding dimension must match embedding_dim")

        attended = self.attention(self.norm1(inputs))
        residual = inputs + attended
        normalized = self.norm2(residual)
        flat = normalized.reshape((batch_size * context_size, embedding_dim))
        hidden = self.feed_forward_in(flat).relu()
        update = self.feed_forward_out(hidden)
        update = update.reshape((batch_size, context_size, embedding_dim))
        return residual + update

    def parameters(self) -> list[Tensor]:
        return [
            *self.norm1.parameters(),
            *self.attention.parameters(),
            *self.norm2.parameters(),
            *self.feed_forward_in.parameters(),
            *self.feed_forward_out.parameters(),
        ]

    def state_dict(self) -> dict:
        return {
            "embedding_dim": self.embedding_dim,
            "context_size": self.context_size,
            "hidden_dim": self.hidden_dim,
            "norm1": self.norm1.state_dict(),
            "attention": self.attention.state_dict(),
            "norm2": self.norm2.state_dict(),
            "feed_forward_in": self.feed_forward_in.state_dict(),
            "feed_forward_out": self.feed_forward_out.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("embedding_dim") != self.embedding_dim:
            raise ValueError("state embedding_dim does not match TransformerBlock")
        if state.get("context_size") != self.context_size:
            raise ValueError("state context_size does not match TransformerBlock")
        if state.get("hidden_dim") != self.hidden_dim:
            raise ValueError("state hidden_dim does not match TransformerBlock")
        self.norm1.load_state_dict(state["norm1"])
        self.attention.load_state_dict(state["attention"])
        self.norm2.load_state_dict(state["norm2"])
        self.feed_forward_in.load_state_dict(state["feed_forward_in"])
        self.feed_forward_out.load_state_dict(state["feed_forward_out"])
