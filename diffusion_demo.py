"""Train the diffusion model (diffusion.py) on MNIST, Fashion-MNIST or CIFAR-10 and draw new images.

    python3 diffusion_demo.py --epochs 2 --out runs/diffusion

Writes the loss log and a grid of generated images (samples.png), each
made by denoising pure random noise in 200 steps.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from time import perf_counter

import kernels
from diffusion import Diffusion, TinyUNet
from features import save_png
from mnist_compare import DATASETS, epoch_batches, load_split
from optim import TensorAdam
from sample_quality import Judge
from tensor import Tensor

_SIGNED = [value / 127.5 - 1.0 for value in range(256)]     # byte -> value in [-1, 1]


def grid_image(values: list[float], count: int, columns: int, shape: tuple[int, int, int] = (1, 28, 28)) -> Tensor:
    """Tile ``count`` images of ``shape`` (values in [-1, 1]) into one picture in [0, 1]:
    (height, width) for grayscale, (3, height, width) for colour."""

    channels, size = shape[0], shape[1]
    rows_of_images = -(-count // columns)
    height, width = rows_of_images * size, columns * size
    pixels = [0.0] * (channels * height * width)
    for index in range(count):
        top, left = (index // columns) * size, (index % columns) * size
        for channel in range(channels):
            image = values[(index * channels + channel) * size * size :]
            plane = channel * height * width
            for r in range(size):
                for c in range(size):
                    value = image[r * size + c]
                    pixels[plane + (top + r) * width + left + c] = min(1.0, max(0.0, (value + 1.0) / 2.0))
    return Tensor(pixels, (height, width) if channels == 1 else (channels, height, width))


class EMA:
    """An exponential moving average of the weights, which DDPM samples from:
    shadow = decay * shadow + (1 - decay) * weights, after every step."""

    def __init__(self, parameters: list[Tensor], decay: float) -> None:
        self.parameters, self.decay = parameters, decay
        self.shadow = [kernels.storage(list(p.data)) for p in parameters]

    def update(self) -> None:
        for index, parameter in enumerate(self.parameters):
            difference = kernels.add_scaled(parameter.data, self.shadow[index], -1.0)
            self.shadow[index] = kernels.add_scaled(self.shadow[index], difference, 1.0 - self.decay)

    def swap(self) -> None:
        """Exchange the live weights with the averaged ones (call again to swap back)."""

        for index, parameter in enumerate(self.parameters):
            parameter.data, self.shadow[index] = self.shadow[index], parameter.data


def sample(process: Diffusion, count: int, args: argparse.Namespace, seed: int) -> list[float]:
    """``count`` images, 64 at a time, with DDIM (--ddim-steps) or DDPM's full chain (0)."""

    values = []
    rng = random.Random(seed)
    for start in range(0, count, 64):
        shape = (min(64, count - start), *DATASETS[args.dataset][1])
        values += (process.ddim_sample(shape, rng, args.ddim_steps) if args.ddim_steps
                   else process.sample(shape, rng))
    return values


def run(args: argparse.Namespace) -> dict:
    kernels.set_default_dtype(args.dtype)
    images, _ = load_split(args.dataset, Path(DATASETS[args.dataset][0]), "train", args.train_limit)
    shape = DATASETS[args.dataset][1]
    model = TinyUNet(channels=shape[0], size=shape[1], width=args.width, groups=args.groups,
                     attention=args.attention, seed=args.seed)
    process = Diffusion(model, steps=args.diffusion_steps)
    optimizer = TensorAdam(model.parameters(), lr=args.lr)
    ema = EMA(model.parameters(), args.ema) if args.ema else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(images)} {args.dataset} images, U-Net with {model.num_parameters()} parameters, "
          f"{args.diffusion_steps} diffusion steps, {args.dtype}", flush=True)
    judge = None
    if args.fid_samples:
        judge = Judge(args.dataset, seed=args.seed)
        print(f"FID classifier: {100 * judge.accuracy:.2f}% test accuracy; "
              f"{args.fid_samples} real training images score {judge.floor(args.fid_samples):.2f}", flush=True)

    rng = random.Random(args.seed)
    log, results = [], {"fid": {}}
    start = perf_counter()
    step = 0
    for epoch in range(1, args.epochs + 1):
        for batch in epoch_batches(len(images), args.batch_size, rng):
            pixels = [_SIGNED[byte] for index in batch for byte in images[index]]
            optimizer.zero_grad()
            loss = process.loss(pixels, (len(batch), *shape), rng)
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update()
            step += 1
            log.append(loss.item())
            if step % args.report_every == 0:
                recent = log[-args.report_every :]
                print(f"epoch {epoch} step {step}: noise-prediction loss {sum(recent) / len(recent):.4f}, "
                      f"{(perf_counter() - start) / 60:.1f} min", flush=True)
        if ema is not None:
            ema.swap()
        model.save(out / "model.pkl")
        if judge is not None and (epoch % args.fid_every == 0 or epoch == args.epochs):
            score = judge.score(sample(process, args.fid_samples, args, args.seed + 1))
            results["fid"][epoch] = score["fid"]
            print(f"epoch {epoch}: FID {score['fid']:.2f} over {args.fid_samples} samples, "
                  f"classes {score['class_counts']}, {(perf_counter() - start) / 60:.1f} min", flush=True)
        if ema is not None:
            ema.swap()

    (out / "loss.txt").write_text("\n".join(f"{value:.6f}" for value in log) + "\n")
    print("sampling...", flush=True)
    if ema is not None:
        ema.swap()
    samples = sample(process, args.samples, args, args.seed + 2)
    save_png(out / "samples.png", grid_image(samples, args.samples, columns=8, shape=shape))
    print(f"wrote {out / 'samples.png'} after {(perf_counter() - start) / 60:.1f} min")
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", choices=("mnist", "fashion", "cifar10"), default="mnist")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--groups", type=int, default=0, help="group norm with this many groups in every block (DDPM: 32)")
    parser.add_argument("--attention", action="store_true", help="self-attention over the middle (7 x 7, CIFAR 8 x 8)")
    parser.add_argument("--diffusion-steps", type=int, default=200)
    parser.add_argument("--ema", type=float, default=0.0, help="decay of an averaged copy of the weights to sample from (DDPM: 0.9999)")
    parser.add_argument("--ddim-steps", type=int, default=0, help="sample with DDIM in this many steps (0: DDPM's full chain)")
    parser.add_argument("--fid-samples", type=int, default=0, help="score this many samples with sample_quality's FID")
    parser.add_argument("--fid-every", type=int, default=1, help="epochs between FID scores")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/diffusion")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
