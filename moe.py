"""Mixture of experts: many feed-forward networks, each token uses a few.

A transformer block's feed-forward network holds most of its weights.
A mixture-of-experts layer (Shazeer et al., 2017; Switch Transformer;
Mixtral) keeps E separate feed-forward "experts" and a small router that
sends every token to its top k of them:

    p = softmax(x W_router)                      one probability per expert
    chosen = the k experts with the largest p
    out = sum over chosen e of  g_e * expert_e(x),   g = p renormalized over the chosen

So the layer has E times the parameters of one feed-forward network, but
each token pays for only k of them. In code, every expert gathers the rows
of the tokens routed to it, runs on that smaller batch, and scatters its
weighted outputs back to those rows.

Left alone, a router tends to favour a few experts, which then get all
the training. The load-balancing loss (Switch Transformer) is minimised
when tokens are spread evenly:

    aux = E * sum over e of  (fraction of tokens sent to e) * (mean p_e)

Training adds a small multiple of it to the language-model loss.
"""

from __future__ import annotations

from tensor import Tensor, gather_rows, scatter_add_rows
from tensor_nn import TensorLinear, TensorModule


class Expert(TensorModule):
    """One feed-forward network: GELU, or SwiGLU (silu(x W_in) * x W_up)."""

    def __init__(self, width: int, hidden: int, *, activation: str = "gelu", seed: int = 0) -> None:
        self.activation = activation
        self.feed_forward_in = TensorLinear(width, hidden, seed=seed)
        self.feed_forward_up = TensorLinear(width, hidden, seed=seed + 1) if activation == "swiglu" else None
        self.feed_forward_out = TensorLinear(hidden, width, seed=seed + 2)

    def __call__(self, x: Tensor) -> Tensor:
        hidden = self.feed_forward_in(x)
        if self.feed_forward_up is not None:
            hidden = hidden * hidden.sigmoid() * self.feed_forward_up(x)
        else:
            hidden = hidden.gelu() if self.activation == "gelu" else hidden.relu()
        return self.feed_forward_out(hidden)

    def parameters(self) -> list[Tensor]:
        layers = [self.feed_forward_in, self.feed_forward_up, self.feed_forward_out]
        return [parameter for layer in layers if layer is not None for parameter in layer.parameters()]


class MoEFeedForward(TensorModule):
    """``num_experts`` experts, ``top_k`` of them per token."""

    def __init__(
        self, width: int, hidden: int, *, num_experts: int = 4, top_k: int = 2,
        activation: str = "gelu", seed: int = 0,
    ) -> None:
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = TensorLinear(width, num_experts, seed=seed)
        self.experts = [Expert(width, hidden, activation=activation, seed=seed + 10 * (e + 1)) for e in range(num_experts)]
        self.aux_loss: Tensor | None = None        # load-balancing loss of the last call

    def __call__(self, x: Tensor) -> Tensor:
        shape, width = x.shape, x.shape[-1]
        count, experts, k = x.numel // width, self.num_experts, self.top_k
        tokens = x.reshape((count, width))

        probabilities = self.router(tokens).softmax(axis=-1)                # (N, E)
        p = probabilities.data
        chosen = [
            sorted(range(experts), key=lambda e: -p[n * experts + e])[:k]   # the routing choice itself has no gradient
            for n in range(count)
        ]
        picked = gather_rows(probabilities.reshape((count * experts, 1)),
                             [n * experts + e for n in range(count) for e in chosen[n]])
        picked = picked.reshape((count, k))
        gates = picked / picked.sum(axis=-1).reshape((count, 1))           # renormalized over the chosen
        gates = gates.reshape((count * k, 1))

        out = None
        for e, expert in enumerate(self.experts):
            slots = [(n, slot) for n in range(count) for slot in range(k) if chosen[n][slot] == e]
            if not slots:
                continue
            rows = [n for n, _ in slots]
            weighted = expert(gather_rows(tokens, rows)) * gather_rows(gates, [n * k + slot for n, slot in slots])
            part = scatter_add_rows(weighted, rows, count)
            out = part if out is None else out + part

        fraction = [0.0] * experts
        for n in range(count):
            for e in chosen[n]:
                fraction[e] += 1.0 / (count * k)
        self.aux_loss = (probabilities.mean(axis=0) * Tensor(fraction, (experts,))).sum() * experts
        return out.reshape(shape)

    def parameters(self) -> list[Tensor]:
        return [*self.router.parameters(), *[p for expert in self.experts for p in expert.parameters()]]

    def state_dict(self) -> dict:
        return {
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "parameters": [list(parameter.data) for parameter in self.parameters()],
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("num_experts") != self.num_experts or state.get("top_k") != self.top_k:
            raise ValueError("state does not match MoEFeedForward")
        parameters = self.parameters()
        if len(state["parameters"]) != len(parameters):
            raise ValueError("state parameters do not match MoEFeedForward")
        for parameter, values in zip(parameters, state["parameters"]):
            if len(values) != parameter.numel:
                raise ValueError("state parameter size does not match MoEFeedForward")
            parameter.data = [float(value) for value in values]
