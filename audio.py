"""Audio: spoken digit recognition from log-mel spectrograms, NanoGrad vs PyTorch.

The Free Spoken Digit Dataset (Jakobovski et al.) is 3,000 recordings of
the digits 0-9: 6 speakers, 50 of each digit each, 8 kHz mono WAV. By the
dataset's convention, recordings 0-4 of every speaker and digit (300) are
the test set and the rest (2,700) train.

Sound is a waveform. Speech models usually start from a spectrogram, which
shows how much energy each frequency has over time:

    frames    cut the signal into 32 ms windows every 10 ms, each tapered by a Hann window
    spectrum  |DFT|^2 of each frame: a matrix product with cosine and sine tables,
              so this runs on the same matmul as everything else
    mel       sum the spectrum into 40 bands spaced like human pitch perception
              (the mel scale: narrow at low frequencies, wide at high ones)
    log       log(energy + 1e-6), since loudness is perceived logarithmically

A clip becomes a 40 x 100 image (shorter clips padded with silence,
longer ones cut at 1 s), and a small CNN classifies it like a picture: three
blocks of 3x3 convolution, batch norm, ReLU and 2x2 pooling, then a
linear layer. Adam at lr 1e-3, batch 32.

    python3 audio.py --epochs 20
    .venv/bin/python audio.py --frameworks nanograd torch --epochs 2

Data: data/fsdd/recordings/*.wav (see data/fsdd/README.md).
"""

from __future__ import annotations

import argparse
import math
import random
import wave
from array import array
from pathlib import Path
from time import perf_counter

from losses import softmax_cross_entropy
from optim import TensorAdam
from tensor import Tensor, no_grad
from tensor_nn import Flatten, MaxPool2d, ReLU, Sequential, TensorBatchNorm2d, TensorConv2D, TensorLinear

RATE, WINDOW, HOP, FFT, BANDS, FRAMES = 8000, 256, 80, 256, 40, 100


def read_wav(path: str | Path) -> list[float]:
    """16-bit mono samples scaled to [-1, 1)."""

    with wave.open(str(path)) as file:
        if file.getsampwidth() != 2 or file.getnchannels() != 1:
            raise ValueError(f"{path}: expected 16-bit mono")
        samples = array("h", file.readframes(file.getnframes()))
    return [s / 32768.0 for s in samples]


def mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_filters(bands: int = BANDS, fft: int = FFT, rate: int = RATE) -> list[list[float]]:
    """Triangular filters (bands x fft // 2 + 1), evenly spaced on the mel scale (HTK's formula)."""

    edges_mel = [mel(0) + (mel(rate / 2) - mel(0)) * i / (bands + 1) for i in range(bands + 2)]
    edges_hz = [700.0 * (10 ** (m / 2595.0) - 1.0) for m in edges_mel]
    bins = [k * rate / fft for k in range(fft // 2 + 1)]
    filters = []
    for b in range(bands):
        low, centre, high = edges_hz[b : b + 3]
        filters.append([max(0.0, min((f - low) / (centre - low), (high - f) / (high - centre))) for f in bins])
    return filters


class Spectrogram:
    """Waveform -> (BANDS, FRAMES) log-mel spectrogram, as two matrix products per clip."""

    def __init__(self) -> None:
        hann = [0.5 - 0.5 * math.cos(2 * math.pi * n / WINDOW) for n in range(WINDOW)]
        half = FFT // 2 + 1
        # Windowed DFT: column k of these holds hann(n) cos(2 pi k n / N) (and sin), so
        # frames @ cos gives the real parts of every frame's spectrum at once.
        self.cos = Tensor([hann[n] * math.cos(2 * math.pi * k * n / FFT) for n in range(WINDOW) for k in range(half)],
                          (WINDOW, half))
        self.sin = Tensor([hann[n] * math.sin(2 * math.pi * k * n / FFT) for n in range(WINDOW) for k in range(half)],
                          (WINDOW, half))
        filters = mel_filters()
        self.mel = Tensor([filters[b][k] for k in range(half) for b in range(BANDS)], (half, BANDS))

    def __call__(self, samples: list[float]) -> list[float]:
        samples = samples[: (FRAMES - 1) * HOP + WINDOW]
        samples = samples + [0.0] * ((FRAMES - 1) * HOP + WINDOW - len(samples))
        frames = Tensor([v for t in range(FRAMES) for v in samples[t * HOP : t * HOP + WINDOW]], (FRAMES, WINDOW))
        with no_grad():
            real, imaginary = frames @ self.cos, frames @ self.sin
            power = real * real + imaginary * imaginary                       # (FRAMES, half)
            energy = (power @ self.mel + 1e-6).log()                          # (FRAMES, BANDS)
        values = energy.data
        return [values[t * BANDS + b] for b in range(BANDS) for t in range(FRAMES)]    # bands as rows


def load_fsdd(folder: str | Path) -> tuple[list[tuple[list[float], int]], list[tuple[list[float], int]]]:
    """(spectrogram, digit) pairs: recordings 5-49 to train, 0-4 to test."""

    spectrogram = Spectrogram()
    train, test = [], []
    for path in sorted(Path(folder).glob("*.wav")):
        digit, _, index = path.stem.split("_")
        (test if int(index) < 5 else train).append((spectrogram(read_wav(path)), int(digit)))
    return train, test


def standardize(train, test) -> None:
    """Zero mean and unit variance per mel band, from the training clips (in place)."""

    for band in range(BANDS):
        values = [v for features, _ in train for v in features[band * FRAMES : (band + 1) * FRAMES]]
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) or 1.0
        for features, _ in train + test:
            features[band * FRAMES : (band + 1) * FRAMES] = [
                (v - mean) / std for v in features[band * FRAMES : (band + 1) * FRAMES]]


def classifier(seed: int = 0) -> Sequential:
    """(B, 1, 40, 100) -> three conv blocks -> (B, 64, 5, 12) -> 10 digit scores."""

    layers, inputs = [], 1
    for block, channels in enumerate((16, 32, 64)):
        layers += [TensorConv2D((channels, inputs, 3, 3), padding=1, use_bias=False, seed=seed + block),
                   TensorBatchNorm2d(channels), ReLU(), MaxPool2d(2)]
        inputs = channels
    return Sequential(*layers, Flatten(), TensorLinear(64 * 5 * 12, 10, seed=seed + 3))


class NanoGradRunner:
    def __init__(self, model: Sequential, lr: float) -> None:
        self.model = model
        self.optimizer = TensorAdam(model.parameters(), lr=lr)

    def train_step(self, features: list[float], labels: list[int]) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        loss = softmax_cross_entropy(self.model(Tensor(features, (len(labels), 1, BANDS, FRAMES))), labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, features: list[float], count: int) -> list[int]:
        self.model.eval()
        with no_grad():
            scores = self.model(Tensor(features, (count, 1, BANDS, FRAMES))).data
        return [max(range(10), key=lambda c: scores[row * 10 + c]) for row in range(count)]

    def weights(self) -> list[list[float]]:
        return [list(p.data) for p in self.model.parameters()]


class TorchRunner:
    def __init__(self, model: Sequential, lr: float) -> None:
        import torch
        from torch_twin import TorchTwin

        self.torch = torch
        self.twin = TorchTwin(model)
        self.optimizer = torch.optim.Adam(self.twin.parameters, lr=lr)

    def train_step(self, features, labels) -> float:
        torch = self.torch
        self.twin.training = True
        self.optimizer.zero_grad()
        scores = self.twin(torch.tensor(features, dtype=torch.float64).reshape(-1, 1, BANDS, FRAMES))
        loss = torch.nn.functional.cross_entropy(scores, torch.tensor(labels))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, features, count: int) -> list[int]:
        torch = self.torch
        self.twin.training = False
        with torch.no_grad():
            return self.twin(torch.tensor(features, dtype=torch.float64).reshape(count, 1, BANDS, FRAMES)
                             ).argmax(1).tolist()

    def weights(self):
        return [p.detach().reshape(-1).tolist() for p in self.twin.parameters]


def accuracy(runner, examples) -> float:
    right = 0
    for first in range(0, len(examples), 100):
        chunk = examples[first : first + 100]
        predicted = runner.predict([v for features, _ in chunk for v in features], len(chunk))
        right += sum(p == label for p, (_, label) in zip(predicted, chunk))
    return right / len(examples)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="data/fsdd/recordings")
    parser.add_argument("--frameworks", nargs="+", choices=("nanograd", "torch"), default=["nanograd"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    start = perf_counter()
    train_set, test_set = load_fsdd(args.data)
    standardize(train_set, test_set)
    print(f"{len(train_set)} training and {len(test_set)} test clips, {BANDS} x {FRAMES} log-mel spectrograms "
          f"({perf_counter() - start:.0f} s)", flush=True)
    results, runners = {}, {}
    for framework in args.frameworks:
        model = classifier(seed=args.seed)
        if framework == "torch":
            import torch

            torch.set_num_threads(1)
            runners[framework] = runner = TorchRunner(model, args.lr)
        else:
            runners[framework] = runner = NanoGradRunner(model, args.lr)
        print(framework, flush=True)
        rng, steps, start = random.Random(args.seed), 0, perf_counter()
        order = list(range(len(train_set)))
        for epoch in range(1, args.epochs + 1):
            rng.shuffle(order)
            total = 0.0
            for first in range(0, len(order), args.batch_size):
                batch = [train_set[i] for i in order[first : first + args.batch_size]]
                total += runner.train_step([v for f, _ in batch for v in f], [label for _, label in batch])
                steps += 1
            print(f"  epoch {epoch:2d}: train loss {total / math.ceil(len(order) / args.batch_size):.4f}, "
                  f"test accuracy {100 * accuracy(runner, test_set):.2f}%, {perf_counter() - start:.0f} s", flush=True)
        results[framework] = {"accuracy": accuracy(runner, test_set),
                              "ms_per_step": 1000 * (perf_counter() - start) / steps}
        print(f"{framework:8s}: test accuracy {100 * results[framework]['accuracy']:.2f}%", flush=True)
    if len(runners) == 2:
        difference = max(abs(a - b) for x, y in zip(runners["nanograd"].weights(), runners["torch"].weights())
                         for a, b in zip(x, y))
        print(f"largest weight difference after training: {difference:.1e}")
        results["max_weight_difference"] = difference
    return results


if __name__ == "__main__":
    main()
