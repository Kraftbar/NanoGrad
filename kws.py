"""Keyword spotting on Speech Commands with a DS-CNN (MLPerf Tiny), NanoGrad vs PyTorch.

Speech Commands v0.02 (Warden, 2018) is 105,829 one-second clips of 35
words, spoken by 2,618 people at 16 kHz. Keyword spotting is the
"Hey Siri" problem in miniature: recognize 10 commands (yes, no, up,
down, left, right, on, off, stop, go), and say "unknown" for any other
word and "silence" for background noise, 12 classes in all. It is
MLPerf Tiny's keyword-spotting benchmark. The test set is the dataset's
official one: 4,890 clips with the 12 classes balanced.

Each clip becomes 49 x 10 MFCCs (MLPerf's front end): 30 ms Hann frames
every 20 ms, a 512-point magnitude spectrum, 40 mel bands from 20 Hz to
4 kHz, a log, and the first 10 coefficients of a DCT (cosine transform)
of each frame's log-mel energies. Both transforms are matrix products.

The model is MLPerf's DS-CNN, a MobileNet for audio: a 10 x 4 stride-2
convolution with 64 filters, four depthwise-separable blocks of 64,
global average pooling and a linear layer (about 23K parameters). The
training follows the reference (Adam, batch 100, lr 5e-4, divided by 5
after each third of training), except that the dropout layers and the
noise and time-shift augmentation are left out. The reference DS-CNN
reaches about 92% (MLPerf Tiny's quality target is 90%).

    python3 kws.py --epochs 24
    .venv/bin/python kws.py --frameworks nanograd torch --epochs 1 --train-limit 2000 --test-limit 500

Data: data/speech_commands/ (see data/speech_commands/DOWNLOAD.md).
"""

from __future__ import annotations

import argparse
import math
import pickle
import random
from pathlib import Path
from time import perf_counter

from audio import mel_filters, read_wav
from losses import softmax_cross_entropy
from mobilenet import MobileNet
from optim import TensorAdam
from tensor import Tensor, no_grad

KEYWORDS = ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")
CLASSES = ("_silence_", "_unknown_") + KEYWORDS
RATE, WINDOW, HOP, FFT, MELS, COEFFICIENTS = 16000, 480, 320, 512, 40, 10
FRAMES = (RATE - WINDOW) // HOP + 1                       # 49


class MFCC:
    """A batch of 1 s waveforms -> (batch, FRAMES * COEFFICIENTS) MFCCs, as matrix products."""

    def __init__(self) -> None:
        half = FFT // 2 + 1
        hann = [0.5 - 0.5 * math.cos(2 * math.pi * n / WINDOW) for n in range(WINDOW)]   # periodic, as TF's
        self.cos = Tensor([hann[n] * math.cos(2 * math.pi * k * n / FFT) for n in range(WINDOW) for k in range(half)],
                          (WINDOW, half))
        self.sin = Tensor([hann[n] * math.sin(2 * math.pi * k * n / FFT) for n in range(WINDOW) for k in range(half)],
                          (WINDOW, half))
        filters = mel_filters(MELS, FFT, RATE, low=20.0, high=4000.0)
        self.mel = Tensor([filters[b][k] for k in range(half) for b in range(MELS)], (half, MELS))
        # Orthonormal DCT-II: coefficient c of a band sequence x is
        # sqrt(2 / M) * sum_m x_m cos(pi c (m + 1/2) / M), with c = 0 scaled by 1 / sqrt(2).
        self.dct = Tensor([math.sqrt(2.0 / MELS) * math.cos(math.pi * c * (m + 0.5) / MELS) * (math.sqrt(0.5) if c == 0
                                                                                          else 1.0)
                           for m in range(MELS) for c in range(COEFFICIENTS)], (MELS, COEFFICIENTS))

    def __call__(self, clips: list[list[float]]) -> list[list[float]]:
        frames = [v for clip in clips for t in range(FRAMES) for v in clip[t * HOP : t * HOP + WINDOW]]
        with no_grad():
            x = Tensor(frames, (len(clips) * FRAMES, WINDOW))
            real, imaginary = x @ self.cos, x @ self.sin
            power = real * real + imaginary * imaginary
            magnitude = power.sqrt()                                                  # |X|
            energy = (magnitude @ self.mel + 1e-6).log()                              # log-mel
            values = (energy @ self.dct).data
        size = FRAMES * COEFFICIENTS
        return [list(values[i * size : (i + 1) * size]) for i in range(len(clips))]


def one_second(samples: list[float]) -> list[float]:
    """Pad with silence (or cut) to exactly 1 s."""

    return (samples + [0.0] * RATE)[:RATE]


def load_speech_commands(folder: str | Path, seed: int = 0) -> dict[str, list[tuple[list[float], int]]]:
    """{"train": [(mfcc, class)], "test": [...]}, computed once and cached. The training set is
    every keyword clip outside the validation and test lists, plus as many again "unknown" words
    and "silence" slices as 10% of the keyword clips each."""

    folder = Path(folder)
    cache = folder / "mfcc.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    held_out = set((folder / "validation_list.txt").read_text().split()) | set(
        (folder / "testing_list.txt").read_text().split())
    rng = random.Random(seed)
    keyword_files, other_files = [], []
    for word_folder in sorted(p for p in folder.iterdir() if p.is_dir() and p.name not in ("_background_noise_", "test_set")):
        for path in sorted(word_folder.glob("*.wav")):
            if f"{word_folder.name}/{path.name}" in held_out:
                continue
            if word_folder.name in KEYWORDS:
                keyword_files.append((path, CLASSES.index(word_folder.name)))
            else:
                other_files.append((path, CLASSES.index("_unknown_")))
    extra = len(keyword_files) // 10
    noise = [read_wav(path) for path in sorted((folder / "_background_noise_").glob("*.wav"))]
    # Each example is read only when its chunk is processed: 40,000 clips of
    # 16,000 Python floats would not fit in memory at once.
    train = [(path, label) for path, label in keyword_files + rng.sample(other_files, extra)]
    for _ in range(extra):                              # silence: a quiet slice of background noise
        source = rng.randrange(len(noise))
        train.append(((source, rng.randrange(len(noise[source]) - RATE), rng.uniform(0.0, 0.1)),
                      CLASSES.index("_silence_")))
    test = [(path, label) for label, name in enumerate(CLASSES)
            for path in sorted((folder / "test_set" / name).glob("*.wav"))]

    def waveform(source) -> list[float]:
        if isinstance(source, Path):
            return one_second(read_wav(source))
        index, offset, volume = source
        return [v * volume for v in noise[index][offset : offset + RATE]]

    mfcc, splits = MFCC(), {}
    for split, examples in (("train", train), ("test", test)):
        features = []
        for first in range(0, len(examples), 200):
            chunk = examples[first : first + 200]
            features += list(zip(mfcc([waveform(source) for source, _ in chunk]), [label for _, label in chunk]))
        splits[split] = features
    cache.write_bytes(pickle.dumps(splits))
    return splits


def ds_cnn(seed: int = 0) -> MobileNet:
    """MLPerf Tiny's DS-CNN: (B, 1, 49, 10) MFCCs -> 12 class scores."""

    return MobileNet(image_shape=(1, FRAMES, COEFFICIENTS), classes=len(CLASSES), stem=64, stem_kernel=(10, 4),
                     stem_stride=2, stem_padding=(5, 1), stages=((64, 1),) * 4, seed=seed)


def standardize(train, test) -> None:
    """Zero mean, unit variance per MFCC coefficient, from the training clips (in place)."""

    for c in range(COEFFICIENTS):
        values = [features[t * COEFFICIENTS + c] for features, _ in train for t in range(FRAMES)]
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) or 1.0
        for features, _ in train + test:
            for t in range(FRAMES):
                features[t * COEFFICIENTS + c] = (features[t * COEFFICIENTS + c] - mean) / std


class NanoGradRunner:
    def __init__(self, model: MobileNet, lr: float) -> None:
        self.model = model
        self.optimizer = TensorAdam(model.parameters(), lr=lr)

    def train_step(self, features: list[float], labels: list[int], lr: float) -> float:
        self.model.train()
        self.optimizer.lr = lr
        self.optimizer.zero_grad()
        loss = softmax_cross_entropy(self.model(Tensor(features, (len(labels), 1, FRAMES, COEFFICIENTS))), labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, features: list[float], count: int) -> list[int]:
        self.model.eval()
        with no_grad():
            scores = self.model(Tensor(features, (count, 1, FRAMES, COEFFICIENTS))).data
        classes = len(CLASSES)
        return [max(range(classes), key=lambda c: scores[row * classes + c]) for row in range(count)]

    def weights(self):
        return [list(p.data) for p in self.model.parameters()]


class TorchRunner:
    def __init__(self, model: MobileNet, lr: float) -> None:
        import torch
        from torch_twin import TorchTwin

        self.torch = torch
        self.twin = TorchTwin(model)
        self.optimizer = torch.optim.Adam(self.twin.parameters, lr=lr)

    def _inputs(self, features):
        return self.torch.tensor(features, dtype=self.torch.float64).reshape(-1, 1, FRAMES, COEFFICIENTS)

    def train_step(self, features, labels, lr: float) -> float:
        torch = self.torch
        self.twin.training = True
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(self.twin(self._inputs(features)), torch.tensor(labels))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, features, count: int) -> list[int]:
        self.twin.training = False
        with self.torch.no_grad():
            return self.twin(self._inputs(features)).argmax(1).tolist()

    def weights(self):
        return [p.detach().reshape(-1).tolist() for p in self.twin.parameters]


def accuracy(runner, examples) -> float:
    right = 0
    for first in range(0, len(examples), 500):
        chunk = examples[first : first + 500]
        predicted = runner.predict([v for f, _ in chunk for v in f], len(chunk))
        right += sum(p == label for p, (_, label) in zip(predicted, chunk))
    return right / len(examples)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="data/speech_commands")
    parser.add_argument("--frameworks", nargs="+", choices=("nanograd", "torch"), default=["nanograd"])
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    start = perf_counter()
    splits = load_speech_commands(args.data, args.seed)
    rng = random.Random(args.seed)
    train_set = [(list(f), label) for f, label in splits["train"]]
    rng.shuffle(train_set)
    train_set, test_set = train_set[: args.train_limit], [(list(f), label) for f, label in splits["test"]]
    rng.shuffle(test_set)
    test_set = test_set[: args.test_limit]
    standardize(train_set, test_set)
    print(f"{len(train_set)} training and {len(test_set)} test clips ({perf_counter() - start:.0f} s)", flush=True)
    results, runners = {}, {}
    for framework in args.frameworks:
        model = ds_cnn(seed=args.seed)
        if framework == "torch":
            import torch

            torch.set_num_threads(1)
            runners[framework] = runner = TorchRunner(model, args.lr)
        else:
            runners[framework] = runner = NanoGradRunner(model, args.lr)
        print(f"{framework} DS-CNN, {model.num_parameters()} parameters", flush=True)
        order, steps, start = list(range(len(train_set))), 0, perf_counter()
        shuffler = random.Random(args.seed + 1)
        for epoch in range(1, args.epochs + 1):
            lr = args.lr / 5 ** min(2, (epoch - 1) * 3 // args.epochs)       # / 5 after each third
            shuffler.shuffle(order)
            total = 0.0
            for first in range(0, len(order), args.batch_size):
                batch = [train_set[i] for i in order[first : first + args.batch_size]]
                total += runner.train_step([v for f, _ in batch for v in f], [label for _, label in batch], lr)
                steps += 1
            print(f"  epoch {epoch:2d}: lr {lr:.0e}, train loss {total / math.ceil(len(order) / args.batch_size):.4f}, "
                  f"{perf_counter() - start:.0f} s", flush=True)
        seconds = perf_counter() - start
        results[framework] = {"accuracy": accuracy(runner, test_set), "ms_per_step": 1000 * seconds / steps}
        print(f"{framework:8s}: test accuracy {100 * results[framework]['accuracy']:.2f}% "
              f"({results[framework]['ms_per_step']:.0f} ms/step)", flush=True)
    if len(runners) == 2:
        difference = max(abs(a - b) for x, y in zip(runners["nanograd"].weights(), runners["torch"].weights())
                         for a, b in zip(x, y))
        print(f"largest weight difference after training: {difference:.1e}")
        results["max_weight_difference"] = difference
    return results


if __name__ == "__main__":
    main()
