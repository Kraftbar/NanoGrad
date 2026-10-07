"""Train the diffusion model (diffusion.py) on MNIST and draw new digits.

    python3 diffusion_demo.py --epochs 2 --out runs/diffusion

Writes the loss log and a grid of generated digits (samples.png), each
made by denoising pure random noise in 200 steps.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from time import perf_counter

from diffusion import Diffusion, TinyUNet
from features import save_png
from mnist_compare import DATASETS, epoch_batches, load_split
from optim import TensorAdam
from tensor import Tensor

_SIGNED = [value / 127.5 - 1.0 for value in range(256)]     # byte -> value in [-1, 1]


def grid_image(values: list[float], count: int, columns: int) -> Tensor:
    """Tile ``count`` 28x28 images (values in [-1, 1]) into one picture in [0, 1]."""

    rows_of_images = -(-count // columns)
    height, width = rows_of_images * 28, columns * 28
    pixels = [0.0] * (height * width)
    for index in range(count):
        top, left = (index // columns) * 28, (index % columns) * 28
        for r in range(28):
            for c in range(28):
                value = values[index * 784 + r * 28 + c]
                pixels[(top + r) * width + left + c] = min(1.0, max(0.0, (value + 1.0) / 2.0))
    return Tensor(pixels, (height, width))


def run(args: argparse.Namespace) -> None:
    images, _ = load_split("mnist", Path(DATASETS["mnist"][0]), "train", args.train_limit)
    model = TinyUNet(width=args.width, seed=args.seed)
    process = Diffusion(model, steps=args.diffusion_steps)
    optimizer = TensorAdam(model.parameters(), lr=args.lr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(images)} images, U-Net with {model.num_parameters()} parameters, "
          f"{args.diffusion_steps} diffusion steps")

    rng = random.Random(args.seed)
    log = []
    start = perf_counter()
    step = 0
    for epoch in range(1, args.epochs + 1):
        for batch in epoch_batches(len(images), args.batch_size, rng):
            pixels = [_SIGNED[byte] for index in batch for byte in images[index]]
            optimizer.zero_grad()
            loss = process.loss(pixels, (len(batch), 1, 28, 28), rng)
            loss.backward()
            optimizer.step()
            step += 1
            log.append(loss.item())
            if step % args.report_every == 0:
                recent = log[-args.report_every :]
                print(f"epoch {epoch} step {step}: noise-prediction loss {sum(recent) / len(recent):.4f}, "
                      f"{(perf_counter() - start) / 60:.1f} min", flush=True)

    (out / "loss.txt").write_text("\n".join(f"{value:.6f}" for value in log) + "\n")
    print("sampling...", flush=True)
    samples = process.sample((args.samples, 1, 28, 28), random.Random(args.seed + 1))
    save_png(out / "samples.png", grid_image(samples, args.samples, columns=8))
    print(f"wrote {out / 'samples.png'} after {(perf_counter() - start) / 60:.1f} min")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--diffusion-steps", type=int, default=200)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/diffusion")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
