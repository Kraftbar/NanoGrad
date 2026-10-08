"""Reusable small language-model modules."""

from __future__ import annotations

import math
import random

from moe import MoEFeedForward
from tensor import Tensor, concat, is_grad_enabled, matmul, rotary_embedding
from tensor_nn import (
    TensorDropout,
    TensorEmbedding,
    TensorLayerNorm,
    TensorLinear,
    TensorModule,
    TensorRMSNorm,
    linear,
)


def recompute(module, *inputs):
    from recompute import recompute as run   # recompute.py builds on this module's layers

    return run(module, *inputs)


TRANSFORMER_INITS = ("xavier", "gpt2")


class TokenPositionEmbedding(TensorModule):
    """Token embeddings plus learned positions for fixed-length contexts."""

    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        embedding_dim: int,
        *,
        learned_positions: bool = True,
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
        # With rotary embeddings (RoPE) positions enter inside attention instead.
        self.position_embedding = (
            TensorEmbedding(context_size, embedding_dim, seed=seed + 1) if learned_positions else None
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
        if self.position_embedding is None:
            return token_vectors

        position_vectors = self.position_embedding(list(range(length)))
        return token_vectors + position_vectors

    def parameters(self) -> list[Tensor]:
        positions = [] if self.position_embedding is None else self.position_embedding.parameters()
        return [*self.token_embedding.parameters(), *positions]

    def state_dict(self) -> dict:
        state = {
            "vocab_size": self.vocab_size,
            "context_size": self.context_size,
            "embedding_dim": self.embedding_dim,
            "token_embedding": self.token_embedding.state_dict(),
        }
        if self.position_embedding is not None:
            state["position_embedding"] = self.position_embedding.state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        if state.get("vocab_size") != self.vocab_size:
            raise ValueError("state vocab_size does not match TokenPositionEmbedding")
        if state.get("context_size") != self.context_size:
            raise ValueError("state context_size does not match TokenPositionEmbedding")
        if state.get("embedding_dim") != self.embedding_dim:
            raise ValueError("state embedding_dim does not match TokenPositionEmbedding")
        self.token_embedding.load_state_dict(state["token_embedding"])
        if self.position_embedding is not None:
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

    ``num_kv_heads`` below ``num_heads`` is grouped-query attention (LLaMA 2
    70B, LLaMA 3, Mistral): each key/value head is shared by a group of
    ``num_heads / num_kv_heads`` query heads, so the KV cache shrinks by that
    factor. ``num_kv_heads=1`` is multi-query attention.

    ``window`` (Mistral's sliding-window attention) lets each token see only
    itself and the ``window - 1`` tokens before it; the KV cache then never
    holds more than ``window`` rows. Stacked layers still pass information
    further back, one window per layer.
    """

    def __init__(
        self,
        embedding_dim: int,
        context_size: int,
        *,
        num_heads: int = 1,
        num_kv_heads: int | None = None,
        dropout: float = 0.0,
        causal: bool = True,
        rotary: bool = False,
        window: int | None = None,
        seed: int = 0,
    ) -> None:
        if window is not None and (window <= 0 or not causal):
            raise ValueError("window must be positive, and needs causal attention")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if context_size <= 0:
            raise ValueError("context_size must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be a positive multiple of num_kv_heads")

        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_heads // num_kv_heads   # query heads per key/value head
        self.causal = causal
        self.rotary = rotary          # rotate queries and keys by position (RoPE)
        self.rope_scale = 1.0         # < 1 after position interpolation (CharTransformerModel.extend_context)
        self.window = window          # sliding-window attention: how far back a token sees
        self.head_dim = embedding_dim // num_heads
        self._mask_cache: dict[int, Tensor] = {}
        self.queries = [
            TensorLinear(embedding_dim, self.head_dim, seed=seed + head * 3)
            for head in range(num_heads)
        ]
        self.keys = [
            TensorLinear(embedding_dim, self.head_dim, seed=seed + head * 3 + 1)
            for head in range(num_kv_heads)
        ]
        self.values = [
            TensorLinear(embedding_dim, self.head_dim, seed=seed + head * 3 + 2)
            for head in range(num_kv_heads)
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

    def __call__(self, inputs: Tensor, *, memory: Tensor | None = None, key_mask: Tensor | None = None) -> Tensor:
        """Attention of ``inputs`` over themselves, or over ``memory``.

        With ``memory`` (B, S, C) this is cross-attention, as in a
        translation model's decoder: queries come from ``inputs``, keys and
        values from ``memory`` (the encoded source sentence), and there is
        no causal mask. ``key_mask`` (B, S), 0 for real keys and -1e9 for
        padding, hides the padding at the end of shorter sequences.
        """

        if inputs.ndim != 3:
            raise ValueError("CausalSelfAttention expects (batch, context, embedding)")

        _, context_size, embedding_dim = inputs.shape
        if context_size > self.context_size:
            raise ValueError("input is longer than context_size")
        if embedding_dim != self.embedding_dim:
            raise ValueError("input embedding dimension must match embedding_dim")

        # Attention runs per sequence: scores are (batch, context, context),
        # so a token only ever looks at tokens from its own row.
        #
        # All heads are computed together. Each head has its own query, key
        # and value layers (D = embedding_dim / heads outputs each); stacking
        # their weights gives one (C x C) projection per role, and reshaping
        # (B, T, H * D) -> (B * H, T, D) turns the heads into extra batch
        # entries. So the whole layer is 3 projections and 2 batched matrix
        # products, instead of 5 small products per head.
        #
        # Grouped-query attention: the G = H / KV query heads that share a
        # key/value head are stacked along the time axis, (B * KV, G * T, D),
        # so the same 2 batched products cover them without copying keys.
        batch, length, _ = inputs.shape
        source = inputs if memory is None else memory
        key_length = source.shape[1]
        heads, kv_heads, group, head_dim = self.num_heads, self.num_kv_heads, self.group_size, self.head_dim
        mask = self._causal_mask(length) if self.causal and memory is None else None
        scale = 1 / math.sqrt(head_dim)

        def project(layers: list[TensorLinear], x: Tensor) -> Tensor:
            steps = x.shape[1]
            weight = concat([layer.weight for layer in layers], axis=0)     # (n * D, C)
            bias = concat([layer.bias for layer in layers], axis=0)         # (n * D,)
            out = linear(x, weight, bias)                                   # (B, T, n * D)
            out = out.reshape((batch, steps, len(layers), head_dim)).permute((0, 2, 1, 3))
            return out.reshape((batch * len(layers), steps, head_dim))      # (B * n, T, D)

        query = project(self.queries, inputs)
        key, value = project(self.keys, source), project(self.values, source)
        if self.rotary and memory is None:    # positions rotate queries and keys, never values
            query = rotary_embedding(query, scale=self.rope_scale)
            key = rotary_embedding(key, scale=self.rope_scale)
        query = query.reshape((batch * kv_heads, group * length, head_dim))  # a no-op when G = 1
        scores = matmul(query, key.permute((0, 2, 1))) * scale             # (B * KV, G * T, S)
        if mask is not None:
            scores = scores + mask                                          # hide future keys
        if key_mask is not None:                                            # hide padding keys
            per_head = [value for row in range(batch) for _ in range(kv_heads)
                        for value in key_mask.data[row * key_length : (row + 1) * key_length]]
            scores = scores + Tensor(per_head, (batch * kv_heads, 1, key_length))
        weights = self.attention_dropout(scores.softmax(axis=-1))
        attended = matmul(weights, value)                                   # (B * KV, G * T, D)
        # Back to (B, T, H * D): each token's head outputs side by side.
        attended = attended.reshape((batch, heads, length, head_dim)).permute((0, 2, 1, 3))
        attended = attended.reshape((batch, length, heads * head_dim))
        return self.output_dropout(self.projection(attended))

    def step(self, inputs: Tensor, cache: dict, position: int = 0) -> Tensor:
        """Attention for one new token, reusing earlier tokens' keys and values.

        ``inputs`` is the new token's (1, 1, C) vector. Its key and value rows
        are appended to ``cache`` (the KV cache), and its query attends over
        every cached row. No mask is needed: only earlier tokens are cached.
        Generating T tokens this way costs one row of work per token, not a
        whole (T x T) attention pass per token.
        """

        scale = 1 / math.sqrt(self.head_dim)
        head_outputs = []
        for kv_head, (key_layer, value_layer) in enumerate(zip(self.keys, self.values)):
            key, value = key_layer(inputs), value_layer(inputs)            # (1, 1, D)
            if self.rotary:
                key = rotary_embedding(key, start=position, scale=self.rope_scale)
            if cache["keys"][kv_head] is not None:
                key = concat([cache["keys"][kv_head], key], axis=1)      # (1, t, D)
                value = concat([cache["values"][kv_head], value], axis=1)
            if self.window is not None and key.shape[1] > self.window:   # a rolling buffer
                key, value = key[:, -self.window :], value[:, -self.window :]
            cache["keys"][kv_head], cache["values"][kv_head] = key, value
            group = self.queries[kv_head * self.group_size : (kv_head + 1) * self.group_size]
            for query_layer in group:                                    # the heads sharing this key/value
                query = query_layer(inputs)
                if self.rotary:
                    query = rotary_embedding(query, start=position, scale=self.rope_scale)
                scores = matmul(query, key.permute((0, 2, 1))) * scale   # (1, 1, t)
                head_outputs.append(matmul(scores.softmax(axis=-1), value))       # (1, 1, D)
        return self.projection(concat(head_outputs, axis=-1))

    def _causal_mask(self, context_size: int) -> Tensor:
        if context_size not in self._mask_cache:
            self._mask_cache[context_size] = _causal_attention_mask(
                context_size, repeats=self.group_size, window=self.window,
            )
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
            "num_kv_heads": self.num_kv_heads,
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
        if state.get("num_kv_heads", self.num_heads) != self.num_kv_heads:
            raise ValueError("state num_kv_heads does not match CausalSelfAttention")
        _load_layer_list(self.queries, state.get("queries"), "queries")
        _load_layer_list(self.keys, state.get("keys"), "keys")
        _load_layer_list(self.values, state.get("values"), "values")
        self.projection.load_state_dict(state["projection"])


def _causal_attention_mask(context_size: int, *, repeats: int = 1, window: int | None = None) -> Tensor:
    """(T, T) additive mask: 0 where a query may look, -1e9 at future keys.

    Example for T = 3 (rows are queries, columns are keys):

        [   0, -1e9, -1e9]
        [   0,    0, -1e9]
        [   0,    0,    0]

    ``repeats`` stacks the rows that many times, (repeats * T, T), for
    query heads stacked along the time axis (grouped-query attention).
    ``window`` also hides keys ``window`` or more positions back.
    """

    reach = context_size if window is None else window
    data = [
        0.0 if query_position - reach < key_position <= query_position else -1e9
        for _ in range(repeats)
        for query_position in range(context_size)
        for key_position in range(context_size)
    ]
    return Tensor(data, (repeats * context_size, context_size))


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
        num_kv_heads: int | None = None,
        feed_forward_activation: str = "relu",
        dropout: float = 0.0,
        causal: bool = True,
        norm: str = "layer",
        rotary: bool = False,
        num_experts: int = 0,
        expert_top_k: int = 2,
        window: int | None = None,
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
        if feed_forward_activation not in FEED_FORWARD_ACTIVATIONS:
            raise ValueError(f"feed_forward_activation must be one of {FEED_FORWARD_ACTIVATIONS}")
        if norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}")

        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.feed_forward_activation = feed_forward_activation
        self.norm_type = norm
        self.norm1 = _norm_layer(norm, embedding_dim)
        self.attention = CausalSelfAttention(
            embedding_dim,
            context_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            causal=causal,
            rotary=rotary,
            window=window,
            seed=seed,
        )
        self.norm2 = _norm_layer(norm, embedding_dim)
        # A mixture of experts (moe.py) replaces the one feed-forward network.
        self.moe = (
            MoEFeedForward(embedding_dim, hidden_dim, num_experts=num_experts, top_k=expert_top_k,
                           activation=feed_forward_activation, seed=seed + 4)
            if num_experts else None
        )
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
        # SwiGLU (LLaMA) multiplies the activated projection by a second,
        # "up" projection: silu(x W_in) * (x W_up).
        self.feed_forward_up = (
            TensorLinear(embedding_dim, hidden_dim, seed=seed + 6)
            if feed_forward_activation == "swiglu" else None
        )
        if self.moe is not None:                      # the experts hold the feed-forward weights
            self.feed_forward_in = self.feed_forward_out = self.feed_forward_up = None
        self.feed_forward_dropout = TensorDropout(dropout, seed=seed + 103)

    def _feed_forward(self, normed: Tensor) -> Tensor:
        if self.moe is not None:
            return self.moe(normed)
        return self.feed_forward_out(self._feed_forward_hidden(normed))

    def _feed_forward_hidden(self, normed: Tensor) -> Tensor:
        hidden = _feed_forward_activation(self.feed_forward_in(normed), self.feed_forward_activation)
        if self.feed_forward_up is not None:
            hidden = hidden * self.feed_forward_up(normed)
        return hidden

    def __call__(self, inputs: Tensor, *, key_mask: Tensor | None = None) -> Tensor:
        """``key_mask`` (B, T): 0 for real tokens, -1e9 for padding (encoders)."""

        if inputs.ndim != 3:
            raise ValueError("TransformerBlock expects (batch, context, embedding)")

        _, context_size, embedding_dim = inputs.shape
        if context_size > self.context_size:
            raise ValueError("input is longer than context_size")
        if embedding_dim != self.embedding_dim:
            raise ValueError("input embedding dimension must match embedding_dim")

        # x = x + attention(norm1(x))
        # x = x + feed_forward(norm2(x))
        attended = self.attention(self.norm1(inputs), key_mask=key_mask)
        residual = inputs + attended
        update = self.feed_forward_dropout(self._feed_forward(self.norm2(residual)))
        return residual + update

    def step(self, inputs: Tensor, cache: dict, position: int = 0) -> Tensor:
        """``__call__`` for one new token, with the attention KV cache."""

        residual = inputs + self.attention.step(self.norm1(inputs), cache, position)
        return residual + self._feed_forward(self.norm2(residual))

    def parameters(self) -> list[Tensor]:
        return [
            *self.norm1.parameters(),
            *self.attention.parameters(),
            *self.norm2.parameters(),
            *self._feed_forward_parameters(),
        ]

    def _feed_forward_parameters(self) -> list[Tensor]:
        if self.moe is not None:
            return self.moe.parameters()
        return [
            *self.feed_forward_in.parameters(),
            *([] if self.feed_forward_up is None else self.feed_forward_up.parameters()),
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
            **self._feed_forward_state(),
        }

    def _feed_forward_state(self) -> dict:
        if self.moe is not None:
            return {"moe": self.moe.state_dict()}
        return {
            "feed_forward_in": self.feed_forward_in.state_dict(),
            "feed_forward_out": self.feed_forward_out.state_dict(),
            **({} if self.feed_forward_up is None else {"feed_forward_up": self.feed_forward_up.state_dict()}),
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
        if ("moe" in state) != (self.moe is not None):
            raise ValueError("state feed-forward (experts or not) does not match TransformerBlock")
        if self.moe is not None:
            self.moe.load_state_dict(state["moe"])
            return
        self.feed_forward_in.load_state_dict(state["feed_forward_in"])
        self.feed_forward_out.load_state_dict(state["feed_forward_out"])
        if self.feed_forward_up is not None:
            self.feed_forward_up.load_state_dict(state["feed_forward_up"])


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
        num_kv_heads: int | None = None,
        num_experts: int = 0,
        expert_top_k: int = 2,
        attention_window: int | None = None,
        recompute_blocks: bool = False,
        num_layers: int = 1,
        feed_forward_activation: str = "relu",
        tie_embeddings: bool = False,
        init: str = "xavier",
        dropout: float = 0.0,
        norm: str = "layer",
        positions: str = "learned",
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
        if feed_forward_activation not in FEED_FORWARD_ACTIVATIONS:
            raise ValueError(f"feed_forward_activation must be one of {FEED_FORWARD_ACTIVATIONS}")
        if positions not in ("learned", "rope"):
            raise ValueError("positions must be 'learned' or 'rope'")

        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.num_experts = num_experts
        self.expert_top_k = expert_top_k
        self.attention_window = attention_window
        self.recompute_blocks = recompute_blocks      # gradient checkpointing (recompute.py)
        self.num_layers = num_layers
        self.feed_forward_activation = feed_forward_activation
        self.tie_embeddings = tie_embeddings
        self.norm_type = norm
        self.positions = positions
        self.embedding = TokenPositionEmbedding(
            vocab_size,
            context_size,
            embedding_dim,
            learned_positions=positions == "learned",
            seed=seed,
        )
        self.blocks = [
            TransformerBlock(
                embedding_dim,
                context_size,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                feed_forward_activation=feed_forward_activation,
                dropout=dropout,
                norm=norm,
                rotary=positions == "rope",
                num_experts=num_experts,
                expert_top_k=expert_top_k,
                window=attention_window,
                seed=seed + 2 + layer_index * 8,
            )
            for layer_index in range(num_layers)
        ]
        self.embedding_dropout = TensorDropout(dropout, seed=seed + 104)
        self.norm = _norm_layer(norm, embedding_dim)
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
            for layer in (
                block.attention.projection,
                *([expert.feed_forward_out for expert in block.moe.experts] if block.moe else [block.feed_forward_out]),
            )
        }
        for parameter in self.parameters():
            if parameter.ndim < 2:
                continue
            std = residual_std if id(parameter) in residual_writers else 0.02
            parameter.data = [rng.gauss(0.0, std) for _ in parameter.data]

    def __call__(self, inputs: Tensor) -> Tensor:
        """Next-token logits after the last position: (batch, vocab)."""

        return self.sequence_logits(inputs)[:, -1]

    def sequence_logits(self, inputs: Tensor) -> Tensor:
        hidden = self.embedding_dropout(self.embedding(inputs))
        for block in self.blocks:
            hidden = recompute(block, hidden) if self.recompute_blocks and is_grad_enabled() else block(hidden)
        return self.projection(self.norm(hidden))

    def extend_context(self, context_size: int, *, interpolate: bool = True) -> None:
        """Accept sequences up to ``context_size`` (rotary positions only).

        Extrapolating runs the new positions with rotation angles the model
        never saw in training. Interpolating (Chen et al., 2023) scales every
        position by old size / new size, so the longer sequence spans the
        trained range of angles, with positions closer together.
        """

        if self.embedding.position_embedding is not None:
            raise ValueError("learned positions cannot be extended; train with positions='rope'")
        scale = self.context_size / context_size if interpolate else 1.0
        for block in self.blocks:
            block.attention.rope_scale = block.attention.rope_scale * scale
            block.attention.context_size = block.context_size = context_size
        self.embedding.context_size = context_size
        self.context_size = context_size

    def auxiliary_loss(self) -> Tensor | None:
        """The experts' load-balancing losses from the last forward pass, summed
        (None without mixture-of-experts layers)."""

        losses = [block.moe.aux_loss for block in self.blocks if block.moe is not None]
        if not losses:
            return None
        total = losses[0]
        for loss in losses[1:]:
            total = total + loss
        return total

    # -- generation with a KV cache (inference only: no dropout) ----------

    def new_cache(self) -> list[dict]:
        """An empty KV cache: per block, per key/value head, the keys and values so far."""

        return [
            {"keys": [None] * self.num_kv_heads, "values": [None] * self.num_kv_heads}
            for _ in self.blocks
        ]

    def step_logits(self, token: int, position: int, cache: list[dict]) -> Tensor:
        """Next-token logits after feeding one more token at ``position``.

        Feeding tokens 0..t one at a time gives the same logits as running
        ``sequence_logits`` on the whole prefix, but each step only computes
        the new token's row (``position`` must stay below context_size).
        """

        hidden = self.embedding.token_embedding([[token]])                     # (1, 1, C)
        if self.embedding.position_embedding is not None:
            hidden = hidden + self.embedding.position_embedding([position])     # + (1, C)
        for block, block_cache in zip(self.blocks, cache):
            hidden = block.step(hidden, block_cache, position)
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
            "num_kv_heads": self.num_kv_heads,
            "num_experts": self.num_experts,
            "expert_top_k": self.expert_top_k,
            "attention_window": self.attention_window,
            "num_layers": self.num_layers,
            "feed_forward_activation": self.feed_forward_activation,
            "tie_embeddings": self.tie_embeddings,
            "norm_type": self.norm_type,
            "positions": self.positions,
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
        if state.get("num_kv_heads", self.num_heads) != self.num_kv_heads:
            raise ValueError("state num_kv_heads does not match CharTransformerModel")
        if (state.get("num_experts", 0), state.get("expert_top_k", 2)) != (self.num_experts, self.expert_top_k):
            raise ValueError("state experts do not match CharTransformerModel")
        if state.get("attention_window") != self.attention_window:
            raise ValueError("state attention_window does not match CharTransformerModel")
        if state.get("num_layers") != self.num_layers:
            raise ValueError("state num_layers does not match CharTransformerModel")
        if state.get("feed_forward_activation", "relu") != self.feed_forward_activation:
            raise ValueError(
                "state feed_forward_activation does not match CharTransformerModel"
            )
        if state.get("tie_embeddings", False) != self.tie_embeddings:
            raise ValueError("state tie_embeddings does not match CharTransformerModel")
        if state.get("norm_type", "layer") != self.norm_type or state.get("positions", "learned") != self.positions:
            raise ValueError("state norm or positions do not match CharTransformerModel")
        blocks = state.get("blocks")
        if not isinstance(blocks, list) or len(blocks) != len(self.blocks):
            raise ValueError("state blocks do not match CharTransformerModel")

        self.embedding.load_state_dict(state["embedding"])
        for block, block_state in zip(self.blocks, blocks):
            block.load_state_dict(block_state)
        self.norm.load_state_dict(state["norm"])
        self.projection.load_state_dict(state["projection"])


FEED_FORWARD_ACTIVATIONS = ("relu", "gelu", "swiglu")
NORMS = ("layer", "rms")


def _norm_layer(norm: str, width: int) -> TensorModule:
    """GPT-2 uses LayerNorm; LLaMA uses RMSNorm."""

    return TensorRMSNorm(width) if norm == "rms" else TensorLayerNorm(width)


def _feed_forward_activation(tensor: Tensor, activation: str) -> Tensor:
    if activation == "relu":
        return tensor.relu()
    if activation == "gelu":
        return tensor.gelu()
    if activation == "swiglu":
        return tensor * tensor.sigmoid()          # SiLU; the "GLU" gate is in the block
    raise ValueError(f"unknown feed-forward activation: {activation}")
