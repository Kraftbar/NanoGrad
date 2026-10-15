"""How much faster is the GPU? One training step of several models on each backend.

    .venv/bin/python gpu_benchmark.py                          # CUDA and C, float32
    .venv/bin/python gpu_benchmark.py --backends cuda --graph  # CUDA, and CUDA with CUDA graphs
    NANOGRAD_THREADS=8 .venv/bin/python gpu_benchmark.py --dtype float64

Each model trains on random data (the step time does not depend on the
values) with its usual optimizer; the time is the mean over ``--steps``
steps after two warm-up steps (the first CUDA step also compiles the
kernels). Every step ends with ``loss.item()``, which waits for the GPU, so
the times are real.

``--graph`` also times each step recorded once as a CUDA graph and then
replayed (``kernels_cuda.capture``): every step copies the batch into the
graph's static inputs, replays it with one launch and reads the loss.
``graphed`` below is the whole recipe.
"""

from __future__ import annotations

import argparse
import random
from time import perf_counter

import kernels
import kernels_cuda
from losses import softmax_cross_entropy
from tensor import Tensor


def graphed(step, *inputs: kernels_cuda.DeviceArray):
    """``step(*inputs)`` (one training step, returning the loss tensor) as a
    CUDA graph: a function of a new batch (one value per input) that copies
    it into ``inputs``, replays the step and returns the loss."""

    step(*inputs)                    # one eager step first: Adam's moments, the index tables, a warm pool
    graph = kernels_cuda.capture(lambda: step(*inputs), *inputs)
    return lambda *batch: graph(*batch).item()


def image_case(model_name: str, image_shape: tuple[int, int, int], batch: int):
    def make(graph: bool):
        import mnist_compare

        model = mnist_compare.build_nanograd(0, model_name, image_shape)
        runner = mnist_compare.NanoGradRunner(model, 1e-3, image_shape=image_shape)
        rng = random.Random(0)
        pixels = [rng.random() for _ in range(batch * image_shape[0] * image_shape[1] * image_shape[2])]
        labels = [rng.randrange(10) for _ in range(batch)]
        if not graph:
            return lambda: runner.train_step(pixels, labels)

        def step(images, targets):   # runner.train_step, returning the loss tensor instead of its value
            runner.optimizer.zero_grad()
            loss = softmax_cross_entropy(model(Tensor(images, (batch, *image_shape))), targets)
            loss.backward()
            runner.optimizer.step()
            return loss

        replay = graphed(step, kernels.storage(pixels), kernels_cuda.DeviceArray("l", labels))
        return lambda: replay(pixels, labels)

    return make


def gpt_case(context: int, width: int, layers: int, heads: int, batch: int):
    def make(graph: bool):
        from nanogpt_compare import CompareConfig, NanoGradRunner

        config = CompareConfig(context_size=context, embedding_dim=width, hidden_dim=4 * width, heads=heads,
                               layers=layers, lr=1e-3, weight_decay=0.1, beta1=0.9, beta2=0.99,
                               max_grad_norm=1.0, batch_size=batch, steps=1, seed=0)
        runner = NanoGradRunner(config, vocab_size=65)
        rng = random.Random(0)
        xs = [[rng.randrange(65) for _ in range(context)] for _ in range(batch)]
        ys = [[rng.randrange(65) for _ in range(context)] for _ in range(batch)]
        if not graph:
            return lambda: runner.train_step(xs, ys)

        def step(tokens, targets):   # runner.train_step, returning the loss tensor
            runner.model.train()
            logits = runner.model.sequence_logits(Tensor(tokens, (batch, context)))
            loss = softmax_cross_entropy(logits.reshape((batch * context, 65)), targets)
            runner.optimizer.zero_grad()
            loss.backward()
            runner.optimizer.step()
            return loss

        tokens, targets = [t for row in xs for t in row], [t for row in ys for t in row]
        replay = graphed(step, kernels.storage(tokens), kernels_cuda.DeviceArray("l", targets))
        return lambda: replay(tokens, targets)

    return make


def diffusion_case(width: int, batch: int, groups: int, attention: bool):
    def make(graph: bool):
        from diffusion import Diffusion, TinyUNet
        from optim import TensorAdam

        if graph:
            return None              # its noise and time steps are drawn on the host, by Python, every step
        model = TinyUNet(width=width, groups=groups, attention=attention, seed=0)
        process = Diffusion(model, steps=1000)
        optimizer = TensorAdam(model.parameters(), lr=5e-4)
        rng = random.Random(0)
        images = [rng.uniform(-1.0, 1.0) for _ in range(batch * 784)]

        def step():
            optimizer.zero_grad()
            loss = process.loss(images, (batch, 1, 28, 28), rng)
            loss.backward()
            optimizer.step()
            return loss.item()

        return step

    return make


CASES = {
    "LeNet-5, MNIST, batch 32": image_case("lenet", (1, 28, 28), 32),
    "small ResNet, MNIST, batch 32": image_case("resnet", (1, 28, 28), 32),
    "ResNet-8, CIFAR-10, batch 32": image_case("resnet8", (3, 32, 32), 32),
    "ResNet-8, CIFAR-10, batch 256": image_case("resnet8", (3, 32, 32), 256),
    "ViT, MNIST, batch 32": image_case("vit", (1, 28, 28), 32),
    "baby GPT (4 x 128, context 64), batch 12": gpt_case(64, 128, 4, 4, 12),
    "GPT (6 x 384, context 256), batch 32": gpt_case(256, 384, 6, 6, 32),
    "diffusion U-Net (width 32, norm + attention), batch 128": diffusion_case(32, 128, 8, True),
}


def time_step(step, steps: int) -> float:
    for _ in range(2):
        step()
    start = perf_counter()
    for _ in range(steps):
        step()
    return (perf_counter() - start) / steps * 1000


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backends", nargs="+", default=["cuda", "c"], choices=("cuda", "c", "python"))
    parser.add_argument("--graph", action="store_true", help="also time CUDA with each step replayed as a CUDA graph")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--only", nargs="*", help="run only the cases whose name contains one of these")
    args = parser.parse_args(argv)

    # Columns: the backends, with "graph" (CUDA, replaying a CUDA graph) after "cuda".
    columns = list(args.backends)
    if args.graph:
        columns.insert(columns.index("cuda") + 1 if "cuda" in columns else 0, "graph")
    results: dict[str, dict[str, float]] = {}
    for name, make in CASES.items():            # model by model, so each model's columns are timed together
        if args.only and not any(part.lower() in name.lower() for part in args.only):
            continue
        for column in columns:
            kernels.use_backend("cuda" if column == "graph" else column)
            kernels.set_default_dtype(args.dtype)
            step = make(graph=column == "graph")
            if step is None:
                print(f"{column:5s} {name}: cannot be recorded as a graph", flush=True)
                continue
            results.setdefault(name, {})[column] = time_step(step, args.steps)
            print(f"{column:5s} {name}: {results[name][column]:.1f} ms", flush=True)

    # The gain: cuda over graph with --graph, else the second column over the first.
    slow, fast = ("cuda", "graph") if args.graph and "cuda" in columns else (columns[-1], columns[0])
    gain = len(columns) == 2 or (slow, fast) == ("cuda", "graph")
    print(f"\n{args.dtype}, ms per training step")
    print("| model | " + " | ".join(columns) + (" | gain |" if gain else " |"))
    print("| --- |" + " ---: |" * (len(columns) + gain))
    for name, times in results.items():
        cells = [f"{times[c]:.1f}" if c in times else "" for c in columns]
        if gain:
            cells.append(f"{times[slow] / times[fast]:.1f}x" if slow in times and fast in times else "")
        print(f"| {name} | " + " | ".join(cells) + " |")
    return results


if __name__ == "__main__":
    main()
