"""Where does the time go? Time per op type, forward and backward.

    from profiler import profile

    with profile() as timings:
        loss = model_loss(...)
        loss.backward()
    print(timings.table())

Backward times are exact: each op's backward function is timed on its own.
Forward times are attributed: every op is charged the time since the
previous op finished, which is its own kernel plus any Python in between
(so a slow loop between two ops shows up on the second one).
"""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter

import tensor


class OpTimings:
    def __init__(self) -> None:
        self.forward_seconds: dict[str, float] = {}
        self.backward_seconds: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self._last = perf_counter()

    def forward(self, op: str) -> None:
        now = perf_counter()
        self.forward_seconds[op] = self.forward_seconds.get(op, 0.0) + (now - self._last)
        self.calls[op] = self.calls.get(op, 0) + 1
        self._last = now

    def backward(self, op: str, seconds: float) -> None:
        self.backward_seconds[op] = self.backward_seconds.get(op, 0.0) + seconds
        self._last = perf_counter()                 # backward time is not the next op's forward

    def table(self, top: int = 15) -> str:
        ops = set(self.forward_seconds) | set(self.backward_seconds)
        total = sum(self.forward_seconds.values()) + sum(self.backward_seconds.values())
        rows = sorted(ops, key=lambda op: -(self.forward_seconds.get(op, 0) + self.backward_seconds.get(op, 0)))
        lines = [f"{'op':24s} {'calls':>6s} {'forward ms':>11s} {'backward ms':>12s} {'share':>6s}"]
        for op in rows[:top]:
            forward, backward = self.forward_seconds.get(op, 0.0), self.backward_seconds.get(op, 0.0)
            lines.append(f"{op:24s} {self.calls.get(op, 0):6d} {1000 * forward:11.1f} {1000 * backward:12.1f} "
                         f"{100 * (forward + backward) / total:5.1f}%")
        lines.append(f"{'total':24s} {'':6s} {1000 * sum(self.forward_seconds.values()):11.1f} "
                     f"{1000 * sum(self.backward_seconds.values()):12.1f}")
        return "\n".join(lines)


@contextmanager
def profile():
    """Time every tensor op run inside the block (one profile at a time)."""

    if tensor._profiler is not None:
        raise RuntimeError("a profile is already running")
    timings = OpTimings()
    tensor._profiler = timings
    try:
        yield timings
    finally:
        tensor._profiler = None
