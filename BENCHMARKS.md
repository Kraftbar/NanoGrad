# Human-Readable Benchmarks

NanoGrad uses small, human-readable benchmarks to keep development grounded.
These checks are meant to answer simple questions:

- Can the scalar engine fit a line?
- Can a tiny classifier separate two classes?
- Can a hidden-layer MLP learn XOR?
- Do gradients still agree with finite differences?

These are not leaderboard benchmarks, production performance claims, or targets
to over-optimize. They are sanity checks that make it easy to see whether the
current system still learns understandable patterns.

## Current Checks

Run the demo:

```bash
python3 demo.py
```

It currently reports:

- Regression loss for fitting a small line-like dataset
- Binary classification loss for separating negative and positive inputs
- XOR loss and final probabilities for a non-linear binary pattern

Run tests:

```bash
python3 -m unittest discover
```

The tests cover scalar gradients, finite-difference checks, regression training,
binary cross entropy, and XOR learning.

## Rule of Thumb

Add a human-readable benchmark when it demonstrates a new capability in a way
that is easy to inspect. Avoid adding benchmarks just to chase numbers before
the underlying implementation is clear and correct.
