"""Learning-rate schedules.

A schedule is a function ``step -> learning rate``. The training loop sets
``optimizer.lr = schedule(step)`` before each optimizer step.

The GPT recipe (nanoGPT's ``get_lr``) has three phases:

    lr
    max |      ________
        |     /        `--.__
        |    /               `-.__
    min |   /                     `--.______________
        +--+-----------------------+-------------->  step
           warmup              decay_steps

1. linear warmup from ~0 to ``max_lr`` over ``warmup_steps``: early
   gradients are large and noisy, and Adam's moment estimates are not
   settled yet, so small steps first
2. cosine decay from ``max_lr`` down to ``min_lr`` until ``decay_steps``
3. constant ``min_lr`` afterwards
"""

from __future__ import annotations

import math
from collections.abc import Callable


Schedule = Callable[[int], float]


def constant(lr: float) -> Schedule:
    """The same learning rate at every step."""

    if lr <= 0.0:
        raise ValueError("lr must be positive")
    return lambda step: lr


def warmup_cosine(
    *,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    decay_steps: int,
) -> Schedule:
    """Linear warmup, then cosine decay to ``min_lr`` (nanoGPT's ``get_lr``).

    Steps count from 0, exactly as in nanoGPT's ``train.py``.
    """

    if not 0.0 <= min_lr <= max_lr or max_lr <= 0.0:
        raise ValueError("need 0 <= min_lr <= max_lr and max_lr > 0")
    if warmup_steps < 0 or decay_steps <= warmup_steps:
        raise ValueError("need 0 <= warmup_steps < decay_steps")

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return max_lr * (step + 1) / (warmup_steps + 1)
        if step > decay_steps:
            return min_lr
        progress = (step - warmup_steps) / (decay_steps - warmup_steps)  # 0 -> 1
        coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))         # 1 -> 0
        return min_lr + coefficient * (max_lr - min_lr)

    return schedule
