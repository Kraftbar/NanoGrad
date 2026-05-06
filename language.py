"""Reusable small language-model modules."""

from __future__ import annotations

import math

from tensor import Tensor, _add_grad, _grad_data, matmul
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
    """Causal self-attention for tiny sequence models."""

    def __init__(
        self,
        embedding_dim: int,
        context_size: int,
        *,
        num_heads: int = 1,
        seed: int = 0,
    ) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if context_size <= 0:
            raise ValueError("context_size must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.queries = [
            TensorLinear(embedding_dim, self.head_dim, seed=seed + head * 3)
            for head in range(num_heads)
        ]
        self.keys = [
            TensorLinear(embedding_dim, self.head_dim, seed=seed + head * 3 + 1)
            for head in range(num_heads)
        ]
        self.values = [
            TensorLinear(embedding_dim, self.head_dim, seed=seed + head * 3 + 2)
            for head in range(num_heads)
        ]
        self.projection = TensorLinear(
            embedding_dim,
            embedding_dim,
            seed=seed + num_heads * 3,
        )

    def __call__(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3:
            raise ValueError("CausalSelfAttention expects (batch, context, embedding)")

        batch_size, context_size, embedding_dim = inputs.shape
        if context_size != self.context_size:
            raise ValueError("input context dimension must match context_size")
        if embedding_dim != self.embedding_dim:
            raise ValueError("input embedding dimension must match embedding_dim")

        flat_inputs = inputs.reshape((batch_size * context_size, embedding_dim))
        mask = _causal_attention_mask(batch_size, context_size)
        head_outputs = []
        for query_layer, key_layer, value_layer in zip(
            self.queries,
            self.keys,
            self.values,
        ):
            query = query_layer(flat_inputs)
            key = key_layer(flat_inputs)
            value = value_layer(flat_inputs)
            scores = matmul(query, key.T) * (1 / math.sqrt(self.head_dim))
            masked_scores = scores + mask
            weights = masked_scores.softmax(axis=1)
            head_outputs.append(matmul(weights, value))

        attended = _concat_feature_tensors(head_outputs)
        output = self.projection(attended)
        return output.reshape((batch_size, context_size, embedding_dim))

    def parameters(self) -> list[Tensor]:
        return [
            *[
                parameter
                for layer in self.queries
                for parameter in layer.parameters()
            ],
            *[
                parameter
                for layer in self.keys
                for parameter in layer.parameters()
            ],
            *[
                parameter
                for layer in self.values
                for parameter in layer.parameters()
            ],
            *self.projection.parameters(),
        ]

    def state_dict(self) -> dict:
        return {
            "embedding_dim": self.embedding_dim,
            "context_size": self.context_size,
            "num_heads": self.num_heads,
            "queries": [
                layer.state_dict()
                for layer in self.queries
            ],
            "keys": [
                layer.state_dict()
                for layer in self.keys
            ],
            "values": [
                layer.state_dict()
                for layer in self.values
            ],
            "projection": self.projection.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("embedding_dim") != self.embedding_dim:
            raise ValueError("state embedding_dim does not match CausalSelfAttention")
        if state.get("context_size") != self.context_size:
            raise ValueError("state context_size does not match CausalSelfAttention")
        if state.get("num_heads") != self.num_heads:
            raise ValueError("state num_heads does not match CausalSelfAttention")
        _load_layer_list(self.queries, state.get("queries"), "queries")
        _load_layer_list(self.keys, state.get("keys"), "keys")
        _load_layer_list(self.values, state.get("values"), "values")
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


def _concat_feature_tensors(tensors: list[Tensor]) -> Tensor:
    if not tensors:
        raise ValueError("feature concatenation needs at least one tensor")
    if len(tensors) == 1:
        return tensors[0]
    if any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError("feature concatenation expects 2D tensors")

    rows = tensors[0].shape[0]
    if any(tensor.shape[0] != rows for tensor in tensors):
        raise ValueError("feature tensors must have the same row count")

    widths = [
        tensor.shape[1]
        for tensor in tensors
    ]
    out_width = sum(widths)
    data = []
    for row in range(rows):
        for tensor, width in zip(tensors, widths):
            start = row * width
            data.extend(tensor.data[start : start + width])

    out = Tensor(
        data,
        (rows, out_width),
        requires_grad=any(tensor.requires_grad for tensor in tensors),
        _children=tuple(tensors),
        _op="concat_features",
    )

    def _backward() -> None:
        grad = _grad_data(out)
        tensor_grads = [
            [0.0] * tensor.numel
            for tensor in tensors
        ]
        for row in range(rows):
            out_offset = row * out_width
            width_offset = 0
            for tensor_index, width in enumerate(widths):
                tensor_offset = row * width
                for col in range(width):
                    tensor_grads[tensor_index][tensor_offset + col] += (
                        grad[out_offset + width_offset + col]
                    )
                width_offset += width

        for tensor, tensor_grad in zip(tensors, tensor_grads):
            if tensor.requires_grad:
                _add_grad(tensor, tensor_grad)

    out._backward = _backward
    return out


def _load_layer_list(layers: list[TensorLinear], state, name: str) -> None:
    if not isinstance(state, list) or len(state) != len(layers):
        raise ValueError(f"state {name} do not match CausalSelfAttention")
    for layer, layer_state in zip(layers, state):
        layer.load_state_dict(layer_state)


class TransformerBlock(TensorModule):
    """Pre-norm transformer block for tiny causal language models."""

    def __init__(
        self,
        embedding_dim: int,
        context_size: int,
        *,
        hidden_dim: int | None = None,
        num_heads: int = 1,
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
        self.num_heads = num_heads
        self.norm1 = TensorLayerNorm(embedding_dim)
        self.attention = CausalSelfAttention(
            embedding_dim,
            context_size,
            num_heads=num_heads,
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
            "num_heads": self.num_heads,
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
        if state.get("num_heads") != self.num_heads:
            raise ValueError("state num_heads does not match TransformerBlock")
        self.norm1.load_state_dict(state["norm1"])
        self.attention.load_state_dict(state["attention"])
        self.norm2.load_state_dict(state["norm2"])
        self.feed_forward_in.load_state_dict(state["feed_forward_in"])
        self.feed_forward_out.load_state_dict(state["feed_forward_out"])


class CharTransformerModel(TensorModule):
    """Tiny causal transformer that predicts the next character after a context."""

    def __init__(
        self,
        vocab_size: int,
        *,
        context_size: int = 4,
        embedding_dim: int = 16,
        hidden_dim: int | None = None,
        num_heads: int = 1,
        num_layers: int = 1,
        seed: int = 0,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if context_size <= 0:
            raise ValueError("context_size must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if hidden_dim is None:
            hidden_dim = embedding_dim * 4
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.embedding = TokenPositionEmbedding(
            vocab_size,
            context_size,
            embedding_dim,
            seed=seed,
        )
        self.blocks = [
            TransformerBlock(
                embedding_dim,
                context_size,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                seed=seed + 2 + layer_index * 8,
            )
            for layer_index in range(num_layers)
        ]
        self.norm = TensorLayerNorm(embedding_dim)
        self.projection = TensorLinear(
            embedding_dim,
            vocab_size,
            seed=seed + 2 + num_layers * 8,
        )

    def __call__(self, inputs: Tensor) -> Tensor:
        return _last_time_step(self.sequence_logits(inputs))

    def sequence_logits(self, inputs: Tensor) -> Tensor:
        hidden = self.embedding(inputs)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.norm(hidden)
        batch_size, context_size, embedding_dim = hidden.shape
        logits = self.projection(
            hidden.reshape((batch_size * context_size, embedding_dim)),
        )
        return logits.reshape((batch_size, context_size, self.vocab_size))

    def parameters(self) -> list[Tensor]:
        block_parameters = [
            parameter
            for block in self.blocks
            for parameter in block.parameters()
        ]
        return [
            *self.embedding.parameters(),
            *block_parameters,
            *self.norm.parameters(),
            *self.projection.parameters(),
        ]

    def state_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "context_size": self.context_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "embedding": self.embedding.state_dict(),
            "blocks": [
                block.state_dict()
                for block in self.blocks
            ],
            "norm": self.norm.state_dict(),
            "projection": self.projection.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("vocab_size") != self.vocab_size:
            raise ValueError("state vocab_size does not match CharTransformerModel")
        if state.get("context_size") != self.context_size:
            raise ValueError("state context_size does not match CharTransformerModel")
        if state.get("embedding_dim") != self.embedding_dim:
            raise ValueError("state embedding_dim does not match CharTransformerModel")
        if state.get("hidden_dim") != self.hidden_dim:
            raise ValueError("state hidden_dim does not match CharTransformerModel")
        if state.get("num_heads") != self.num_heads:
            raise ValueError("state num_heads does not match CharTransformerModel")
        if state.get("num_layers") != self.num_layers:
            raise ValueError("state num_layers does not match CharTransformerModel")
        blocks = state.get("blocks")
        if not isinstance(blocks, list) or len(blocks) != len(self.blocks):
            raise ValueError("state blocks do not match CharTransformerModel")

        self.embedding.load_state_dict(state["embedding"])
        for block, block_state in zip(self.blocks, blocks):
            block.load_state_dict(block_state)
        self.norm.load_state_dict(state["norm"])
        self.projection.load_state_dict(state["projection"])


def _last_time_step(sequence: Tensor) -> Tensor:
    if sequence.ndim != 3:
        raise ValueError("last time step expects a 3D tensor")

    batch_size, context_size, width = sequence.shape
    data = []
    for batch in range(batch_size):
        start = (batch * context_size + context_size - 1) * width
        data.extend(sequence.data[start : start + width])

    out = Tensor(
        data,
        (batch_size, width),
        requires_grad=sequence.requires_grad,
        _children=(sequence,),
        _op="last_time_step",
    )

    def _backward() -> None:
        if not sequence.requires_grad:
            return

        grad = _grad_data(out)
        sequence_grad = [0.0] * sequence.numel
        for batch in range(batch_size):
            source_start = (batch * context_size + context_size - 1) * width
            grad_start = batch * width
            for offset in range(width):
                sequence_grad[source_start + offset] += grad[grad_start + offset]
        _add_grad(sequence, sequence_grad)

    out._backward = _backward
    return out
