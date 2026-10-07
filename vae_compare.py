"""Train the VAE (vae.py) on MNIST in NanoGrad and in PyTorch.

Both start from the same weights and see the same batches *and the same
random noise* for the reparameterization trick, so in float64 they compute
the same thing. The script reports the test-set loss (negative ELBO, in
nats per image, lower is better) after every epoch, and writes a grid of
digits decoded from random codes as a PGM image.

    .venv/bin/python vae_compare.py --epochs 3
    python3 vae_compare.py --frameworks nanograd --samples-pgm runs/vae_samples.pgm
"""

from __future__ import annotations

import argparse
import importlib.util
import random
from pathlib import Path
from time import perf_counter

from features import save_pgm
from mnist_compare import DATASETS, batch_pixels, epoch_batches, load_split
from optim import TensorAdam
from tensor import Tensor, no_grad
from vae import VAE, vae_loss


FRAMEWORKS = ("nanograd", "torch")


class NanoGradRunner:
    def __init__(self, model: VAE, lr: float) -> None:
        self.model = model
        self.optimizer = TensorAdam(model.parameters(), lr=lr)

    def train_step(self, pixels, noise, count) -> float:
        self.optimizer.zero_grad()
        images = Tensor(pixels, (count, self.model.pixels))
        logits, mean, log_var = self._forward(images, noise, count)
        loss, _, _ = vae_loss(logits, images, mean, log_var)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _forward(self, images, noise, count):
        logits, mean, log_var = self.model(images, Tensor(noise, (count, self.model.latent)))
        return logits, mean, log_var

    def evaluate(self, pixels, noise, count) -> tuple[float, float, float]:
        with no_grad():
            images = Tensor(pixels, (count, self.model.pixels))
            logits, mean, log_var = self._forward(images, noise, count)
            loss, reconstruction, kl = vae_loss(logits, images, mean, log_var)
        return loss.item(), reconstruction, kl

    def decode(self, codes, count) -> list[float]:
        with no_grad():
            logits = self.model.decode(Tensor(codes, (count, self.model.latent)))
            return list(logits.sigmoid().data)

    def weights(self) -> list:
        return [parameter.data for parameter in self.model.parameters()]


class TorchRunner:
    """The same VAE in PyTorch, layer by layer through torch_twin.py."""

    def __init__(self, model: VAE, lr: float, threads: int) -> None:
        import torch

        from torch_twin import TorchTwin

        torch.set_num_threads(threads)
        self.torch = torch
        self.model = model
        self.twin = TorchTwin(model)
        self.optimizer = torch.optim.Adam(self.twin.parameters, lr=lr)

    def _loss(self, pixels, noise, count):
        torch, twin, model = self.torch, self.twin, self.model
        images = torch.tensor(pixels, dtype=torch.float64).reshape(count, model.pixels)
        hidden = torch.relu(twin.forward(model.encoder, images))
        mean, log_var = twin.forward(model.to_mean, hidden), twin.forward(model.to_log_var, hidden)
        codes = mean + torch.exp(log_var / 2) * torch.tensor(noise, dtype=torch.float64).reshape(count, model.latent)
        logits = twin.forward(model.decoder_out, torch.relu(twin.forward(model.decoder_hidden, codes)))
        reconstruction = torch.nn.functional.binary_cross_entropy_with_logits(logits, images, reduction="sum") / count
        kl = -0.5 * torch.sum(1 + log_var - mean ** 2 - torch.exp(log_var)) / count
        return reconstruction + kl, reconstruction, kl

    def train_step(self, pixels, noise, count) -> float:
        self.optimizer.zero_grad()
        loss, _, _ = self._loss(pixels, noise, count)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, pixels, noise, count) -> tuple[float, float, float]:
        with self.torch.no_grad():
            loss, reconstruction, kl = self._loss(pixels, noise, count)
        return loss.item(), reconstruction.item(), kl.item()

    def weights(self) -> list:
        return [parameter.detach().reshape(-1).tolist() for parameter in self.twin.parameters]


def gaussian(count: int, rng: random.Random) -> list[float]:
    return [rng.gauss(0.0, 1.0) for _ in range(count)]


def run(args: argparse.Namespace) -> dict:
    data_dir = Path(DATASETS["mnist"][0])
    train_images, _ = load_split("mnist", data_dir, "train", args.train_limit)
    test_images, _ = load_split("mnist", data_dir, "test", args.test_limit)
    model = VAE(latent=args.latent, hidden=args.hidden, seed=args.seed)
    print(f"train {len(train_images)} images, test {len(test_images)}, latent {args.latent}, "
          f"{model.num_parameters()} parameters")

    runners = {}
    for name in args.frameworks:
        if name == "nanograd":
            runners[name] = NanoGradRunner(model, args.lr)
        else:
            if importlib.util.find_spec("torch") is None:
                raise SystemExit("PyTorch is not installed; use --frameworks nanograd")
            runners[name] = TorchRunner(model, args.lr, args.torch_threads)   # copies the initial weights

    rng = random.Random(args.seed)
    schedule = [epoch_batches(len(train_images), args.batch_size, rng) for _ in range(args.epochs)]
    noise_rng = random.Random(args.seed + 1)
    noise = [[gaussian(len(batch) * args.latent, noise_rng) for batch in batches] for batches in schedule]
    test_indices = list(range(len(test_images)))
    test_noise = gaussian(len(test_images) * args.latent, random.Random(args.seed + 2))

    results = {}
    for name, runner in runners.items():
        rows = []
        for epoch, batches in enumerate(schedule, start=1):
            start = perf_counter()
            for batch, batch_noise in zip(batches, noise[epoch - 1]):
                runner.train_step(batch_pixels(train_images, batch), batch_noise, len(batch))
            seconds = perf_counter() - start
            loss, reconstruction, kl = runner.evaluate(
                batch_pixels(test_images, test_indices), test_noise, len(test_indices),
            )
            rows.append({"epoch": epoch, "test_loss": loss, "reconstruction": reconstruction, "kl": kl,
                         "ms_per_step": 1000 * seconds / len(batches)})
            print(f"{name:8s} epoch {epoch}: test loss {loss:.2f} nats/image "
                  f"(reconstruction {reconstruction:.2f} + KL {kl:.2f}), "
                  f"{1000 * seconds / len(batches):.1f} ms/step", flush=True)
        results[name] = rows

    if len(runners) == 2:
        difference = max(
            abs(a - b)
            for left, right in zip(runners["nanograd"].weights(), runners["torch"].weights())
            for a, b in zip(left, right)
        )
        print(f"max |weight difference| after training: {difference:.3e}")
        results["max_weight_difference"] = difference

    if args.samples_pgm and "nanograd" in runners:
        # A grid of digits decoded from random codes z ~ normal(0, 1).
        grid = args.grid
        images = runners["nanograd"].decode(gaussian(grid * grid * args.latent, random.Random(7)), grid * grid)
        rows = []
        for row in range(grid * 28):
            for column in range(grid * 28):
                index = (row // 28) * grid + column // 28
                rows.append(images[index * 784 + (row % 28) * 28 + column % 28])
        Path(args.samples_pgm).parent.mkdir(parents=True, exist_ok=True)
        save_pgm(args.samples_pgm, Tensor(rows, (grid * 28, grid * 28)))
        print(f"wrote {args.samples_pgm}")
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--frameworks", nargs="+", choices=FRAMEWORKS, default=list(FRAMEWORKS))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--samples-pgm", help="write decoded random samples to this PGM file")
    parser.add_argument("--grid", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
