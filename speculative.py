"""Speculative decoding: a small model drafts, the large model checks.

Generating with a large model costs one full forward pass per token. In
speculative decoding (Leviathan et al., 2023; Chen et al., 2023) a small,
cheap "draft" model proposes the next k tokens, and the large "target"
model scores all of them in a single forward pass. Each proposed token x,
drafted with probability q(x), is accepted with probability

    min(1, p(x) / q(x))                p = the target model's distribution

and at the first rejection the token is redrawn from the leftover
distribution  max(0, p - q), renormalized. This rule makes every token
come out distributed exactly as if the target had sampled it alone, so
the speed-up is free of approximation. If all k are accepted, the
target's pass also gives the next token for free: up to k + 1 tokens per
target pass, and more the better the draft agrees with the target.

With ``temperature=0`` both models pick their most likely token, and the
rule reduces to "accept while the draft's choice equals the target's".
"""

from __future__ import annotations

import math
import random

from tensor import Tensor, no_grad
from train import evaluation_mode


def _distributions(model, tokens: list[int], count: int, context_size: int, temperature: float) -> list[list[float]]:
    """The model's next-token distributions after each of the last ``count`` positions."""

    window = tokens[-context_size:]
    with no_grad(), evaluation_mode(model):
        logits = model.sequence_logits(Tensor.from_list([window]))
    vocab = logits.shape[-1]
    rows = [list(logits.data[position * vocab : (position + 1) * vocab]) for position in range(len(window))]
    return [_probabilities(row, temperature) for row in rows[-count:]]


def _probabilities(logits: list[float], temperature: float) -> list[float]:
    if temperature == 0.0:                         # greedy: all mass on the most likely token
        best = max(range(len(logits)), key=logits.__getitem__)
        return [1.0 if index == best else 0.0 for index in range(len(logits))]
    top = max(logits)
    weights = [math.exp((value - top) / temperature) for value in logits]
    total = sum(weights)
    return [weight / total for weight in weights]


def _sample(probabilities: list[float], rng: random.Random) -> int:
    return rng.choices(range(len(probabilities)), weights=probabilities)[0]


def speculative_generate(
    target, draft, prompt: list[int], *, length: int, context_size: int,
    lookahead: int = 4, temperature: float = 1.0, rng: random.Random,
) -> tuple[list[int], dict]:
    """``length`` new tokens after ``prompt``; also returns counts of target
    passes and accepted draft tokens."""

    if lookahead < 1:
        raise ValueError("lookahead must be at least 1")
    if lookahead >= context_size:
        raise ValueError("lookahead must be shorter than the context")
    tokens = list(prompt)
    stats = {"target_passes": 0, "drafted": 0, "accepted": 0}
    while len(tokens) - len(prompt) < length:
        k = min(lookahead, length - (len(tokens) - len(prompt)))
        proposal, drafts = [], []
        for _ in range(k):                                    # 1. the draft proposes k tokens
            q = _distributions(draft, tokens + proposal, 1, context_size, temperature)[0]
            proposal.append(_sample(q, rng))
            drafts.append(q)
        # 2. one target pass scores all k proposals (and the position after them)
        targets = _distributions(target, tokens + proposal, k + 1, context_size, temperature)
        stats["target_passes"] += 1
        stats["drafted"] += k
        for position, token in enumerate(proposal):           # 3. accept or correct
            p, q = targets[position], drafts[position]
            if rng.random() < min(1.0, p[token] / q[token]):
                tokens.append(token)
                stats["accepted"] += 1
                continue
            leftover = [max(0.0, a - b) for a, b in zip(p, q)]
            tokens.append(_sample(leftover if sum(leftover) > 0 else p, rng))
            break
        else:                                                 # all accepted: one more from the target
            tokens.append(_sample(targets[k], rng))
    return tokens[: len(prompt) + length], stats
