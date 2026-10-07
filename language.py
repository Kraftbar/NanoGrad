"""Reusable small language-model modules."""

from __future__ import annotations

import math
import random

from tensor import Tensor, _add_grad, _grad_data, _op_result, _set_backward, concat, matmul
from tensor_nn import (
    TensorDropout,
    TensorEmbedding,
    TensorLayerNorm,
    TensorLinear,
    TensorModule,
)


TRANSFORMER_INITS = ("xavier", "gpt2")


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
        """Token vectors plus position vectors for positions 0 .. T - 1.

        Any length T up to ``context_size`` works (like nanoGPT): a shorter
        input simply uses the first T learned positions.
        """

        token_vectors = self.token_embedding(indices)
        length = token_vectors.shape[-2]
        if length > self.context_size:
            raise ValueError("input is longer than context_size")

        position_vectors = self.position_embedding(list(range(length)))
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
        # The projection reads every position's features at once, so this
        # model needs exactly context_size tokens.
        if inputs.shape[-1] != self.context_size:
            raise ValueError("input last dimension must match context_size")
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
    """Multi-head self-attention, causal by default.

    Causal (GPT): a token only attends to itself and earlier tokens.
    With ``causal=False`` every token attends to every token, as in a
    vision transformer or a BERT-style encoder.
    """

    def __init__(
        self,
        embedding_dim: int,
        context_size: int,
        *,
        num_heads: int = 1,
        dropout: float = 0.0,
        causal: bool = True,
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
        self.causal = causal
        self.head_dim = embedding_dim // num_heads
        self._mask_cache: dict[int, Tensor] = {}
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
        # nanoGPT's two attention dropouts: on the attention weights, and on
        # the output before it joins the residual stream.
        self.attention_dropout = TensorDropout(dropout, seed=seed + 101)
        self.output_dropout = TensorDropout(dropout, seed=seed + 102)

    def __call__(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3:
            raise ValueError("CausalSelfAttention expects (batch, context, embedding)")

        _, context_size, embedding_dim = inputs.shape
        if context_size > self.context_size:
            raise ValueError("input is longer than context_size")
        if embedding_dim != self.embedding_dim:
            raise ValueError("input embedding dimension must match embedding_dim")

        # Attention runs per sequence: scores are (batch, context, context),
        # so a token only ever looks at tokens from its own row.
        mask = self._causal_mask(context_size) if self.causal else None
        scale = 1 / math.sqrt(self.head_dim)
        head_outputs = []
        for query_layer, key_layer, value_layer in zip(
            self.queries,
            self.keys,
            self.values,
        ):
            query = query_layer(inputs)                      # (B, T, D)
            key = key_layer(inputs)                          # (B, T, D)
            value = value_layer(inputs)                      # (B, T, D)
            key_t = key.permute((0, 2, 1))                   # (B, D, T)
            scores = matmul(query, key_t) * scale            # (B, T, T)
            if mask is not None:
                scores = scores + mask                       # hide future keys
            weights = scores.softmax(axis=-1)                # (B, T, T)
            weights = self.attention_dropout(weights)
            head_outputs.append(matmul(weights, value))      # (B, T, D)

        attended = concat(head_outputs, axis=-1)             # (B, T, C)
        return self.output_dropout(self.projection(attended))

    def step(self, inputs: Tensor, cache: dict) -> Tensor:
        """Attention for one new token, reusing earlier tokens' keys and values.

        ``inputs`` is the new token's (1, 1, C) vector. Its key and value rows
        are appended to ``cache`` (the KV cache), and its query attends over
        every cached row. No mask is needed: only earlier tokens are cached.
        Generating T tokens this way costs one row of work per token, not a
        whole (T x T) attention pass per token.
        """

        scale = 1 / math.sqrt(self.head_dim)
        head_outputs = []
        for head, (query_layer, key_layer, value_layer) in enumerate(
            zip(self.queries, self.keys, self.values)
        ):
            key, value = key_layer(inputs), value_layer(inputs)          # (1, 1, D)
            if cache["keys"][head] is not None:
                key = concat([cache["keys"][head], key], axis=1)         # (1, t, D)
                value = concat([cache["values"][head], value], axis=1)
            cache["keys"][head], cache["values"][head] = key, value
            scores = matmul(query_layer(inputs), key.permute((0, 2, 1))) * scale   # (1, 1, t)
            head_outputs.append(matmul(scores.softmax(axis=-1), value))           # (1, 1, D)
        return self.projection(concat(head_outputs, axis=-1))

    def _causal_mask(self, context_size: int) -> Tensor:
        if context_size not in self._mask_cache:
            self._mask_cache[context_size] = _causal_attention_mask(context_size)
        return self._mask_cache[context_size]

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


def _causal_attention_mask(context_size: int) -> Tensor:
    """(T, T) additive mask: 0 where a query may look, -1e9 at future keys.

    Example for T = 3 (rows are queries, columns are keys):

        [   0, -1e9, -1e9]
        [   0,    0, -1e9]
        [   0,    0,    0]
    """

    data = [
        0.0 if key_position <= query_position else -1e9
        for query_position in range(context_size)
        for key_position in range(context_size)
    ]
    return Tensor(data, (context_size, context_size))


def _load_layer_list(layers: list[TensorLinear], state, name: str) -> None:
    if not isinstance(state, list) or len(state) != len(layers):
        raise ValueError(f"state {name} do not match CausalSelfAttention")
    for layer, layer_state in zip(layers, state):
        layer.load_state_dict(layer_state)


class TransformerBlock(TensorModule):
    """Pre-norm transformer block (causal for GPT, or not: see attention)."""

    def __init__(
        self,
        embedding_dim: int,
        context_size: int,
        *,
        hidden_dim: int | None = None,
        num_heads: int = 1,
        feed_forward_activation: str = "relu",
        dropout: float = 0.0,
        causal: bool = True,
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
        if feed_forward_activation not in ("relu", "gelu"):
            raise ValueError("feed_forward_activation must be 'relu' or 'gelu'")

        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.feed_forward_activation = feed_forward_activation
        self.norm1 = TensorLayerNorm(embedding_dim)
        self.attention = CausalSelfAttention(
            embedding_dim,
            context_size,
            num_heads=num_heads,
            dropout=dropout,
            causal=causal,
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
        self.feed_forward_dropout = TensorDropout(dropout, seed=seed + 103)

    def __call__(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3:
            raise ValueError("TransformerBlock expects (batch, context, embedding)")

        _, context_size, embedding_dim = inputs.shape
        if context_size > self.context_size:
            raise ValueError("input is longer than context_size")
        if embedding_dim != self.embedding_dim:
            raise ValueError("input embedding dimension must match embedding_dim")

        # x = x + attention(norm1(x))
        # x = x + feed_forward(norm2(x))
        attended = self.attention(self.norm1(inputs))
        residual = inputs + attended
        hidden = _feed_forward_activation(
            self.feed_forward_in(self.norm2(residual)),
            self.feed_forward_activation,
        )
        update = self.feed_forward_dropout(self.feed_forward_out(hidden))
        return residual + update

    def step(self, inputs: Tensor, cache: dict) -> Tensor:
        """``__call__`` for one new token, with the attention KV cache."""

        residual = inputs + self.attention.step(self.norm1(inputs), cache)
        hidden = _feed_forward_activation(
            self.feed_forward_in(self.norm2(residual)),
            self.feed_forward_activation,
        )
        return residual + self.feed_forward_out(hidden)

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
            "feed_forward_activation": self.feed_forward_activation,
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
        if state.get("feed_forward_activation", "relu") != self.feed_forward_activation:
            raise ValueError(
                "state feed_forward_activation does not match TransformerBlock"
            )
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
        feed_forward_activation: str = "relu",
        tie_embeddings: bool = False,
        init: str = "xavier",
        dropout: float = 0.0,
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
        if init not in TRANSFORMER_INITS:
            raise ValueError(f"init must be one of {TRANSFORMER_INITS}")
        if feed_forward_activation not in ("relu", "gelu"):
            raise ValueError("feed_forward_activation must be 'relu' or 'gelu'")

        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.feed_forward_activation = feed_forward_activation
        self.tie_embeddings = tie_embeddings
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
                feed_forward_activation=feed_forward_activation,
                dropout=dropout,
                seed=seed + 2 + layer_index * 8,
            )
            for layer_index in range(num_layers)
        ]
        self.embedding_dropout = TensorDropout(dropout, seed=seed + 104)
        self.norm = TensorLayerNorm(embedding_dim)
        self.projection = TensorLinear(
            embedding_dim,
            vocab_size,
            weight=(
                self.embedding.token_embedding.weight
                if tie_embeddings
                else None
            ),
            seed=seed + 2 + num_layers * 8,
        )
        if init == "gpt2":
            self._gpt2_init(random.Random(seed))

    def _gpt2_init(self, rng: random.Random) -> None:
        """Re-draw weights the way GPT-2 / nanoGPT initialize them.

        - every linear weight and embedding table ~ normal(0, 0.02)
        - biases 0, LayerNorm scale 1 and shift 0 (already true)
        - projections that write into the residual stream (attention output
          and feed-forward output) ~ normal(0, 0.02 / sqrt(2 * num_layers)),
          so the residual sum does not grow with depth
        """

        residual_std = 0.02 / math.sqrt(2 * self.num_layers)
        residual_writers = {
            id(layer.weight)
            for block in self.blocks
            for layer in (block.attention.projection, block.feed_forward_out)
        }
        for parameter in self.parameters():
            if parameter.ndim < 2:
                continue
            std = residual_std if id(parameter) in residual_writers else 0.02
            parameter.data = [rng.gauss(0.0, std) for _ in parameter.data]

    def __call__(self, inputs: Tensor) -> Tensor:
        return _last_time_step(self.sequence_logits(inputs))

    def sequence_logits(self, inputs: Tensor) -> Tensor:
        hidden = self.embedding_dropout(self.embedding(inputs))
        for block in self.blocks:
            hidden = block(hidden)
        return self.projection(self.norm(hidden))

    # -- generation with a KV cache (inference only: no dropout) ----------

    def new_cache(self) -> list[dict]:
        """An empty KV cache: per block, per head, the keys and values so far."""

        return [
            {"keys": [None] * self.num_heads, "values": [None] * self.num_heads}
            for _ in self.blocks
        ]

    def step_logits(self, token: int, position: int, cache: list[dict]) -> Tensor:
        """Next-token logits after feeding one more token at ``position``.

        Feeding tokens 0..t one at a time gives the same logits as running
        ``sequence_logits`` on the whole prefix, but each step only computes
        the new token's row (``position`` must stay below context_size).
        """

        token_vector = self.embedding.token_embedding([[token]])               # (1, 1, C)
        hidden = token_vector + self.embedding.position_embedding([position])   # + (1, C)
        for block, block_cache in zip(self.blocks, cache):
            hidden = block.step(hidden, block_cache)
        return self.projection(self.norm(hidden)).reshape((self.vocab_size,))

    def parameters(self) -> list[Tensor]:
        block_parameters = [
            parameter
            for block in self.blocks
            for parameter in block.parameters()
        ]
        projection_parameters = (
            [self.projection.bias]
            if self.tie_embeddings
            else self.projection.parameters()
        )
        return [
            *self.embedding.parameters(),
            *block_parameters,
            *self.norm.parameters(),
            *projection_parameters,
        ]

    def state_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "context_size": self.context_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "feed_forward_activation": self.feed_forward_activation,
            "tie_embeddings": self.tie_embeddings,
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
        if state.get("feed_forward_activation", "relu") != self.feed_forward_activation:
            raise ValueError(
                "state feed_forward_activation does not match CharTransformerModel"
            )
        if state.get("tie_embeddings", False) != self.tie_embeddings:
            raise ValueError("state tie_embeddings does not match CharTransformerModel")
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

    out = _op_result(data, (batch_size, width), (sequence,), "last_time_step")

    def _backward() -> None:
        grad = _grad_data(out)
        sequence_grad = [0.0] * sequence.numel
        for batch in range(batch_size):
            source_start = (batch * context_size + context_size - 1) * width
            grad_start = batch * width
            for offset in range(width):
                sequence_grad[source_start + offset] += grad[grad_start + offset]
        _add_grad(sequence, sequence_grad)

    _set_backward(out, _backward)
    return out


def _feed_forward_activation(tensor: Tensor, activation: str) -> Tensor:
    if activation == "relu":
        return tensor.relu()
    if activation == "gelu":
        return tensor.gelu()
    raise ValueError(f"unknown feed-forward activation: {activation}")
