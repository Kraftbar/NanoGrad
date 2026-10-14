"""How much faster is the GPU? One training step of several models on each backend.

    .venv/bin/python gpu_benchmark.py                          # CUDA and C, float32
    NANOGRAD_THREADS=8 .venv/bin/python gpu_benchmark.py --dtype float64

Each model trains on random data (the step time does not depend on the
values) with its usual optimizer; the time is the mean over ``--steps``
steps after two warm-up steps (the first CUDA step also compiles the
kernels). Every step ends with ``loss.item()``, which waits for the GPU, so
the times are real.
"""

from __future__ import annotations

import argparse
import random
from time import perf_counter

import kernels


def image_case(model_name: str, image_shape: tuple[int, int, int], batch: int):
    def make():
        import mnist_compare

        model = mnist_compare.build_nanograd(0, model_name, image_shape)
        runner = mnist_compare.NanoGradRunner(model, 1e-3, image_shape=image_shape)
        rng = random.Random(0)
        pixels = [rng.random() for _ in range(batch * image_shape[0] * image_shape[1] * image_shape[2])]
        labels = [rng.randrange(10) for _ in range(batch)]
        return lambda: runner.train_step(pixels, labels)

    return make


def gpt_case(context: int, width: int, layers: int, heads: int, batch: int):
    def make():
        from nanogpt_compare import CompareConfig, NanoGradRunner

        config = CompareConfig(context_size=context, embedding_dim=width, hidden_dim=4 * width, heads=heads,
                               layers=layers, lr=1e-3, weight_decay=0.1, beta1=0.9, beta2=0.99,
                               max_grad_norm=1.0, batch_size=batch, steps=1, seed=0)
        runner = NanoGradRunner(config, vocab_size=65)
        rng = random.Random(0)
        xs = [[rng.randrange(65) for _ in range(context)] for _ in range(batch)]
        ys = [[rng.randrange(65) for _ in range(context)] for _ in range(batch)]
        return lambda: runner.train_step(xs, ys)

    return make


def diffusion_case(width: int, batch: int, groups: int, attention: bool):
    def make():
        from diffusion import Diffusion, TinyUNet
        from optim import TensorAdam

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
    "nanoGPT Shakespeare (6 x 384, context 256), batch 16": gpt_case(256, 384, 6, 6, 16),
    "diffusion U-Net (width 32, norm + attention), batch 128": diffusion_case(32, 128, 8, True),
}


def time_step(make, steps: int) -> float:
    step = make()
    for _ in range(2):
        step()
    start = perf_counter()
    for _ in range(steps):
        step()
    return (perf_counter() - start) / steps * 1000


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backends", nargs="+", default=["cuda", "c"], choices=("cuda", "c", "python"))
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--only", nargs="*", help="run only the cases whose name contains one of these")
    args = parser.parse_args(argv)

    results: dict[str, dict[str, float]] = {}
    for backend in args.backends:
        kernels.use_backend(backend)
        kernels.set_default_dtype(args.dtype)
        for name, make in CASES.items():
            if args.only and not any(part.lower() in name.lower() for part in args.only):
                continue
            results.setdefault(name, {})[backend] = time_step(make, args.steps)
            print(f"{backend:5s} {name}: {results[name][backend]:.1f} ms", flush=True)

    print(f"\n{args.dtype}, ms per training step")
    print("| model | " + " | ".join(args.backends) + (" | speedup |" if len(args.backends) == 2 else " |"))
    print("| --- |" + " ---: |" * (len(args.backends) + (len(args.backends) == 2)))
    for name, times in results.items():
        cells = [f"{times[b]:.1f}" if b in times else "" for b in args.backends]
        if len(args.backends) == 2 and all(b in times for b in args.backends):
            cells.append(f"{times[args.backends[1]] / times[args.backends[0]]:.1f}x")
        print(f"| {name} | " + " | ".join(cells) + " |")
    return results


if __name__ == "__main__":
    main()
