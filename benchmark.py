"""Time NanoGrad's core ops, forward + backward, against PyTorch.

In the spirit of Baidu's DeepBench: instead of a whole model, time the
individual operations models are built from, at the shapes real models use
(LeNet, ResNet and ViT in ``mnist_compare.py``, the baby GPT in
``gpt_train.py``). Each op runs a forward pass and a backward pass with a
fixed random output gradient.

    python3 benchmark.py                       # NanoGrad, C and Python kernels
    .venv/bin/python benchmark.py --torch      # plus PyTorch, and check agreement

With ``--torch``, PyTorch runs in float64 on one thread (NanoGrad's setting),
and every NanoGrad output and input gradient is checked against PyTorch's.
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from statistics import median
from time import perf_counter

import kernels
from losses import softmax_cross_entropy
from tensor import Tensor, avg_pool2d, conv2d, conv2d_valid, matmul, max_pool2d
from tensor_nn import TensorBatchNorm2d, layer_norm, linear


@dataclass(frozen=True)
class Case:
    """One op: input shapes, and the op written for NanoGrad and for PyTorch."""

    name: str
    shapes: tuple[tuple[int, ...], ...]
    nanograd: Callable[..., Tensor]
    torch: Callable[..., object]
    targets: tuple[int, int] | None = None   # (count, classes) for a loss
    model: str = field(default="")


CASES = [
    Case("matmul 768x128 @ 128x512", ((768, 128), (128, 512)),
         lambda a, b: matmul(a, b), lambda a, b: a @ b, model="GPT"),
    Case("linear 768x128 -> 512", ((768, 128), (512, 128), (512,)),
         lambda x, w, b: linear(x, w, b), lambda x, w, b: x @ w.T + b, model="GPT"),
    Case("layer_norm 768x128", ((768, 128), (128,), (128,)),
         lambda x, w, b: layer_norm(x, w, b),
         lambda x, w, b: _torch().nn.functional.layer_norm(x, (128,), w, b), model="GPT"),
    Case("gelu 768x512", ((768, 512),),
         lambda x: x.gelu(), lambda x: _torch().nn.functional.gelu(x), model="GPT"),
    Case("softmax 12x4x64x64", ((12, 4, 64, 64),),
         lambda x: x.softmax(), lambda x: x.softmax(-1), model="GPT"),
    Case("cross_entropy 768x65", ((768, 65),),
         lambda x, targets: softmax_cross_entropy(x, targets),
         lambda x, targets: _torch().nn.functional.cross_entropy(x, targets),
         targets=(768, 65), model="GPT"),
    Case("conv2d 32x1x28x28 * 6x1x5x5", ((32, 1, 28, 28), (6, 1, 5, 5)),
         lambda x, k: conv2d_valid(x, k), lambda x, k: _torch().nn.functional.conv2d(x, k), model="LeNet"),
    Case("conv2d 32x6x12x12 * 16x6x5x5", ((32, 6, 12, 12), (16, 6, 5, 5)),
         lambda x, k: conv2d_valid(x, k), lambda x, k: _torch().nn.functional.conv2d(x, k), model="LeNet"),
    Case("tanh 32x6x24x24", ((32, 6, 24, 24),),
         lambda x: x.tanh(), lambda x: x.tanh(), model="LeNet"),
    Case("avg_pool 2x2 32x6x24x24", ((32, 6, 24, 24),),
         lambda x: avg_pool2d(x, (2, 2)), lambda x: _torch().nn.functional.avg_pool2d(x, 2), model="LeNet"),
    Case("max_pool 2x2 32x6x24x24", ((32, 6, 24, 24),),
         lambda x: max_pool2d(x, (2, 2)), lambda x: _torch().nn.functional.max_pool2d(x, 2), model="LeNet"),
    Case("conv2d 3x3 pad 1 32x8x28x28 * 8x8x3x3", ((32, 8, 28, 28), (8, 8, 3, 3)),
         lambda x, k: conv2d(x, k, padding=1),
         lambda x, k: _torch().nn.functional.conv2d(x, k, padding=1), model="ResNet"),
    Case("conv2d 3x3 stride 2 32x8x28x28 * 16x8x3x3", ((32, 8, 28, 28), (16, 8, 3, 3)),
         lambda x, k: conv2d(x, k, stride=2, padding=1),
         lambda x, k: _torch().nn.functional.conv2d(x, k, stride=2, padding=1), model="ResNet"),
    Case("batch_norm 32x8x28x28", ((32, 8, 28, 28), (8,), (8,)),
         lambda x, w, b: _batch_norm(x, w, b),
         lambda x, w, b: _torch().nn.functional.batch_norm(x, None, None, w, b, training=True), model="ResNet"),
    Case("relu 32x8x28x28", ((32, 8, 28, 28),),
         lambda x: x.relu(), lambda x: x.relu(), model="ResNet"),
]


# ---------------------------------------------------------------------------
# Running one case
# ---------------------------------------------------------------------------


def case_data(case: Case, seed: int = 0) -> tuple[list[list[float]], list[int] | None]:
    rng = random.Random(seed)
    inputs = [[rng.uniform(-1.0, 1.0) for _ in range(_numel(shape))] for shape in case.shapes]
    targets = None
    if case.targets is not None:
        count, classes = case.targets
        targets = [rng.randrange(classes) for _ in range(count)]
    return inputs, targets


def nanograd_step(case: Case, inputs, targets, out_grad=None) -> Callable[[], tuple]:
    """Build the inputs once; the returned function runs forward + backward.

    It returns (output, input gradients). Without ``out_grad`` the output
    gradient is all ones.
    """

    tensors = [Tensor(list(values), shape, requires_grad=True) for values, shape in zip(inputs, case.shapes)]
    extra = (targets,) if targets is not None else ()

    # Like PyTorch's side, the output gradient is built once, outside the timing.
    seed = None if out_grad is None else Tensor(out_grad, (len(out_grad),))

    def step():
        for tensor in tensors:   # as PyTorch's side does: each run starts with no gradient
            tensor.grad = None
        out = case.nanograd(*tensors, *extra)
        out.backward(seed.reshape(out.shape) if seed is not None else [1.0] * out.numel)
        return out.data, [tensor.grad for tensor in tensors]

    return step


def torch_step(case: Case, inputs, targets, out_grad) -> Callable[[], tuple]:
    torch = _torch()
    tensors = [
        torch.tensor(values, dtype=torch.float64).reshape(shape).requires_grad_()
        for values, shape in zip(inputs, case.shapes)
    ]
    extra = (torch.tensor(targets),) if targets is not None else ()
    grad = torch.tensor(out_grad, dtype=torch.float64)

    def step():
        for tensor in tensors:
            tensor.grad = None
        out = case.torch(*tensors, *extra)
        out.backward(grad.reshape(out.shape))
        return out, [tensor.grad for tensor in tensors]

    return step


def _as_lists(result) -> tuple[list[float], list[list[float]]]:
    out, grads = result
    return out.detach().reshape(-1).tolist(), [grad.reshape(-1).tolist() for grad in grads]


def output_gradient(case: Case, inputs, targets) -> list[float]:
    """A fixed random gradient for the op's output (shape found by one run)."""

    with_ones, _ = nanograd_step(case, inputs, targets)()
    rng = random.Random(1)
    return [rng.uniform(-1.0, 1.0) for _ in with_ones]


def time_ms(function: Callable[[], object], repeat: int) -> float:
    """Median wall time in milliseconds over ``repeat`` runs, after a warmup."""

    function()
    times = []
    for _ in range(repeat):
        start = perf_counter()
        function()
        times.append(perf_counter() - start)
    return 1000.0 * median(times)


def max_difference(left: list[list[float]], right: list[list[float]]) -> float:
    return max(abs(a - b) for x, y in zip(left, right) for a, b in zip(x, y))


# ---------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> list[dict]:
    backends = ["c", "python"] if kernels.BACKEND == "c" else ["python"]
    if args.torch:
        _torch().set_num_threads(1)

    rows = []
    for case in CASES:
        if args.only and not any(word in case.name for word in args.only):
            continue
        inputs, targets = case_data(case)
        grad = output_gradient(case, inputs, targets)
        row = {"op": case.name, "model": case.model}
        for backend in backends:
            kernels.use_backend(backend)
            row[f"nanograd_{backend}_ms"] = time_ms(nanograd_step(case, inputs, targets, grad), args.repeat)
        kernels.use_backend(backends[0])
        if args.torch:
            step = torch_step(case, inputs, targets, grad)
            row["torch_ms"] = time_ms(step, args.repeat)
            ours, theirs = nanograd_step(case, inputs, targets, grad)(), _as_lists(step())
            row["max_difference"] = max(max_difference([ours[0]], [theirs[0]]), max_difference(ours[1], theirs[1]))
        rows.append(row)
        print(_format(row), flush=True)
    return rows


def _format(row: dict) -> str:
    parts = [f"{row['op']:30s}"]
    for key, label in (("nanograd_c_ms", "C"), ("nanograd_python_ms", "Python"), ("torch_ms", "PyTorch")):
        if key in row:
            parts.append(f"{label} {row[key]:8.2f} ms")
    if "torch_ms" in row and "nanograd_c_ms" in row:
        parts.append(f"C/PyTorch {row['nanograd_c_ms'] / row['torch_ms']:5.1f}x")
    if "max_difference" in row:
        parts.append(f"max diff {row['max_difference']:.1e}")
    return "  ".join(parts)


def average_seconds(function: Callable[[], None], *, repeat: int) -> float:
    if repeat <= 0:
        raise ValueError("repeat must be positive")

    start = perf_counter()
    for _ in range(repeat):
        function()
    return (perf_counter() - start) / repeat


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--torch", action="store_true", help="also time PyTorch and check agreement")
    parser.add_argument("--only", nargs="+", help="run ops whose name contains one of these words")
    return parser.parse_args(argv)


def _batch_norm(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    norm = TensorBatchNorm2d(weight.shape[0])
    norm.weight, norm.bias = weight, bias
    return norm(x)


def _torch():
    import torch

    return torch


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
