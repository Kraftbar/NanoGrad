"""Gradient checkpointing: keep fewer activations, recompute them for backward.

Backpropagation needs every intermediate result of the forward pass, so
training memory grows with depth: for a transformer, each block keeps its
attention scores, softmax outputs and feed-forward activations until
backward reaches it. Gradient checkpointing (Chen et al., 2016) keeps only
each block's *input*. The block runs forward without recording anything,
and when backward arrives at it, the block runs forward again (this time
recording) and backpropagates through that fresh copy. The cost is one
extra forward pass of the block, about a third more compute, for memory
that no longer grows with what is inside the blocks.

    out = recompute(block, x)          # instead of out = block(x)

The recomputed forward pass must repeat the original exactly, so the
random generators of the block's dropout layers are rewound to where they
were the first time.
"""

from __future__ import annotations

from tensor import Tensor, _add_grad, _grad_data, _no_backward, _op_result, _set_backward, no_grad
from tensor_nn import TensorDropout, TensorModule


def recompute(module: TensorModule, *inputs: Tensor) -> Tensor:
    """``module(*inputs)``, without keeping the module's intermediate results."""

    parameters = module.parameters()
    dropouts = [m for m in module.modules() if isinstance(m, TensorDropout)]
    random_states = [dropout.rng.getstate() for dropout in dropouts]
    with no_grad():
        result = module(*inputs)
    out = _op_result(result.data, result.shape, (*inputs, *parameters), "recompute")

    def _backward() -> None:
        for dropout, state in zip(dropouts, random_states):      # the same masks again
            dropout.rng.setstate(state)
        copies = [Tensor.__new__(Tensor) for _ in inputs]
        for copy, original in zip(copies, inputs):               # cut off from the outer graph
            copy.data, copy.shape = original.data, original.shape
            copy.requires_grad, copy.grad = original.requires_grad, None
            copy._prev, copy._op, copy._backward = set(), "", _no_backward
        again = module(*copies)
        # Parameters add their share to whatever they already hold.
        again.backward(_grad_data(out), accumulate=True)
        for copy, original in zip(copies, inputs):
            if original.requires_grad:
                _add_grad(original, copy.grad)

    _set_backward(out, _backward)
    return out
