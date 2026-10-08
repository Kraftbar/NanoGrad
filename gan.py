"""A deep convolutional GAN (DCGAN) that learns to draw MNIST digits.

A generative adversarial network (Goodfellow et al., 2014) trains two
networks against each other:

    generator G:      random noise z -> an image
    discriminator D:  an image -> a score for "this is a real image"

D learns to tell real images from G's; G learns to make D call its images
real. Each step:

    1. D: binary cross entropy pushing D(real) -> 1 and D(G(z)) -> 0
    2. G: binary cross entropy pushing D(G(z)) -> 1 (the "non-saturating" loss)

DCGAN (Radford et al., 2016) is the convolutional recipe of PyTorch's
``examples/dcgan``: G grows the image with transposed convolutions
(7x7 -> 14x14 -> 28x28 here), D shrinks it with strided convolutions,
batch norm in both, ReLU in G and LeakyReLU(0.2) in D, a tanh output,
weights from normal(0, 0.02), Adam at lr 2e-4 with beta1 0.5.

    python3 gan.py --epochs 5 --out runs/gan
    .venv/bin/python gan.py --compare --steps 10     # same weights and noise in PyTorch
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from time import perf_counter

from losses import binary_cross_entropy_with_logits
from mnist_compare import load_split
from optim import TensorAdam
from tensor import Tensor, no_grad
from tensor_nn import (
    Flatten, LeakyReLU, ReLU, Sequential, Tanh, TensorBatchNorm2d, TensorConv2D, TensorConvTranspose2D,
    TensorModule,
)
from train import evaluation_mode


def generator(noise: int = 64, width: int = 32, seed: int = 0) -> Sequential:
    """z (B, noise, 1, 1) -> image (B, 1, 28, 28) in [-1, 1]."""

    return Sequential(
        TensorConvTranspose2D(noise, 2 * width, 7, use_bias=False, seed=seed),            # 1 -> 7
        TensorBatchNorm2d(2 * width), ReLU(),
        TensorConvTranspose2D(2 * width, width, 4, stride=2, padding=1, use_bias=False, seed=seed + 1),  # 14
        TensorBatchNorm2d(width), ReLU(),
        TensorConvTranspose2D(width, 1, 4, stride=2, padding=1, use_bias=False, seed=seed + 2),          # 28
        Tanh(),
    )


def discriminator(width: int = 32, seed: int = 0) -> Sequential:
    """image (B, 1, 28, 28) -> one logit per image (B, 1)."""

    return Sequential(
        TensorConv2D((width, 1, 4, 4), stride=2, padding=1, use_bias=False, seed=seed),          # 28 -> 14
        LeakyReLU(0.2),
        TensorConv2D((2 * width, width, 4, 4), stride=2, padding=1, use_bias=False, seed=seed + 1),  # -> 7
        TensorBatchNorm2d(2 * width), LeakyReLU(0.2),
        TensorConv2D((1, 2 * width, 7, 7), use_bias=False, seed=seed + 2),                       # -> 1
        Flatten(),
    )


def dcgan_init(model: TensorModule, rng: random.Random) -> None:
    """The DCGAN paper's initialization: conv weights ~ normal(0, 0.02),
    batch-norm scales ~ normal(1, 0.02), shifts 0."""

    for module in model.modules():
        if isinstance(module, (TensorConv2D, TensorConvTranspose2D)):
            module.kernel.data = [rng.gauss(0.0, 0.02) for _ in range(module.kernel.numel)]
        elif isinstance(module, TensorBatchNorm2d):
            module.weight.data = [rng.gauss(1.0, 0.02) for _ in range(module.weight.numel)]


class GAN:
    def __init__(self, args) -> None:
        rng = random.Random(args.seed)
        self.noise = args.noise
        self.generator, self.discriminator = generator(args.noise, args.width, args.seed), discriminator(args.width, args.seed + 10)
        dcgan_init(self.generator, rng)
        dcgan_init(self.discriminator, rng)
        settings = dict(lr=args.lr, beta1=0.5, beta2=0.999)
        self.g_optimizer = TensorAdam(self.generator.parameters(), **settings)
        self.d_optimizer = TensorAdam(self.discriminator.parameters(), **settings)

    def train_step(self, real: list[float], z: list[float], batch: int) -> tuple[float, float]:
        images = Tensor(real, (batch, 1, 28, 28))
        noise = Tensor(z, (batch, self.noise, 1, 1))
        ones, zeros = Tensor([1.0] * batch, (batch, 1)), Tensor([0.0] * batch, (batch, 1))

        # 1. discriminator: real -> 1, fake -> 0 (the fakes are constants here)
        self.d_optimizer.zero_grad()
        fake = self.generator(noise)
        d_loss = (binary_cross_entropy_with_logits(self.discriminator(images), ones)
                  + binary_cross_entropy_with_logits(self.discriminator(fake.detach()), zeros))
        d_loss.backward()
        self.d_optimizer.step()

        # 2. generator: make the discriminator say 1 for the fakes
        self.g_optimizer.zero_grad()
        g_loss = binary_cross_entropy_with_logits(self.discriminator(fake), ones)
        g_loss.backward()
        self.g_optimizer.step()
        return d_loss.item(), g_loss.item()

    def sample(self, z: list[float], count: int) -> list[float]:
        with no_grad(), evaluation_mode(self.generator):
            return list(self.generator(Tensor(z, (count, self.noise, 1, 1))).data)


def to_unit_range(pixels) -> list[float]:
    return [value / 127.5 - 1.0 for value in pixels]          # [0, 255] -> [-1, 1], tanh's range


def batches(images: list[bytes], batch: int, rng: random.Random):
    """Full batches of shuffled images, each batch one flat list of pixels."""

    order = list(range(len(images)))
    rng.shuffle(order)
    for start in range(0, len(order) - batch + 1, batch):
        yield [value for index in order[start : start + batch] for value in images[index]]


def save_grid(pixels: list[float], count: int, path: Path) -> None:
    from features import save_png

    side = int(count ** 0.5)
    grid = [0.0] * (side * 28) ** 2
    for index in range(side * side):
        top, left = (index // side) * 28, (index % side) * 28
        for row in range(28):
            for col in range(28):
                value = (pixels[index * 784 + row * 28 + col] + 1.0) / 2.0
                grid[(top + row) * side * 28 + left + col] = min(1.0, max(0.0, value))
    save_png(path, Tensor(grid, (side * 28, side * 28)))


def run(args) -> None:
    images, _ = load_split("mnist", Path(args.data), "train", args.train_limit)
    gan = GAN(args)
    rng = random.Random(args.seed + 1)
    fixed = [rng.gauss(0.0, 1.0) for _ in range(64 * args.noise)]           # the same noise for every grid
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    for epoch in range(1, args.epochs + 1):
        losses = []
        for real in batches(images, args.batch_size, rng):
            z = [rng.gauss(0.0, 1.0) for _ in range(args.batch_size * args.noise)]
            losses.append(gan.train_step(to_unit_range(real), z, args.batch_size))
        d, g = (sum(pair[i] for pair in losses) / len(losses) for i in (0, 1))
        print(f"epoch {epoch}: discriminator loss {d:.3f}, generator loss {g:.3f}, "
              f"{(perf_counter() - start) / 60:.1f} min", flush=True)
        save_grid(gan.sample(fixed, 64), 64, out / f"samples_epoch{epoch}.png")
    print(f"wrote {out}/samples_epoch*.png")


def compare(args) -> None:
    """Both frameworks from the same weights, on the same images and noise."""

    import torch

    from torch_twin import TorchTwin

    torch.set_num_threads(1)
    images, _ = load_split("mnist", Path(args.data), "train", args.steps * args.batch_size)
    gan = GAN(args)
    g_twin, d_twin = TorchTwin(gan.generator), TorchTwin(gan.discriminator)
    rng = random.Random(args.seed + 1)
    steps = []
    for real in batches(images, args.batch_size, rng):
        steps.append((to_unit_range(real), [rng.gauss(0.0, 1.0) for _ in range(args.batch_size * args.noise)]))
    b, bce = args.batch_size, torch.nn.functional.binary_cross_entropy_with_logits
    g_opt = torch.optim.Adam(g_twin.parameters, lr=args.lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(d_twin.parameters, lr=args.lr, betas=(0.5, 0.999))

    start = perf_counter()
    ours = [gan.train_step(real, z, b) for real, z in steps]
    ours_ms = 1000 * (perf_counter() - start) / len(steps)
    start, theirs = perf_counter(), []
    for real, z in steps:
        images_t = torch.tensor(real, dtype=torch.float64).reshape(b, 1, 28, 28)
        noise_t = torch.tensor(z, dtype=torch.float64).reshape(b, args.noise, 1, 1)
        d_opt.zero_grad()
        fake = g_twin(noise_t)
        d_loss = bce(d_twin(images_t), torch.ones(b, 1, dtype=torch.float64)) + bce(
            d_twin(fake.detach()), torch.zeros(b, 1, dtype=torch.float64))
        d_loss.backward()
        d_opt.step()
        g_opt.zero_grad()
        g_loss = bce(d_twin(fake), torch.ones(b, 1, dtype=torch.float64))
        g_loss.backward()
        g_opt.step()
        theirs.append((d_loss.item(), g_loss.item()))
    theirs_ms = 1000 * (perf_counter() - start) / len(steps)
    print(f"nanograd: {ours_ms:.1f} ms/step, losses {ours[-1][0]:.6f} {ours[-1][1]:.6f}")
    print(f"torch   : {theirs_ms:.1f} ms/step, losses {theirs[-1][0]:.6f} {theirs[-1][1]:.6f}")
    difference = max(abs(a - c) for model, twin in ((gan.generator, g_twin), (gan.discriminator, d_twin))
                     for p, t in zip(model.parameters(), twin.parameters)
                     for a, c in zip(p.data, t.detach().reshape(-1).tolist()))
    print(f"largest weight difference after {len(steps)} steps: {difference:.1e}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="data/mnist")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--noise", type=int, default=64)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--out", default="runs/gan")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    compare(args) if args.compare else run(args)


if __name__ == "__main__":
    main()
