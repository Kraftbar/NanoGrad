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
after each third of training; the reference trains 36 epochs). Its dropout
layers, 0.2 after the first convolution and 0.4 before the pooling, are
``--dropout``; with it the PyTorch twin draws different masks, so the two
frameworks' weights no longer agree. The reference DS-CNN reaches about
92% (MLPerf Tiny's quality target is 90%).

The reference's code asks for random time shifts and background noise,
but draws them with NumPy inside the function it hands to tf.data's
``map``, which runs once, when the graph is built: its shift is a
constant 2-sample pad-and-slice (no shift), and every training clip gets
the same noise slice at the same volume. What it does have is more data:
TensorFlow Datasets' training split labels every other word's clip
"unknown" (85,511 clips; ``--all-unknown``). ``--augment`` is real
augmentation, as in TensorFlow's original speech_commands example: each
training clip is shifted and mixed with noise afresh every epoch (see
``augmented``), and its MFCCs are recomputed; without it they are
computed once and cached.

    python3 kws.py --epochs 24 [--augment] [--all-unknown] [--dropout]
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

from array import array

from audio import mel_filters, read_wav
from losses import softmax_cross_entropy
from mobilenet import MobileNet
from optim import TensorAdam
from tensor import Tensor, no_grad

KEYWORDS = ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")
CLASSES = ("_silence_", "_unknown_") + KEYWORDS
RATE, WINDOW, HOP, FFT, MELS, COEFFICIENTS = 16000, 480, 320, 512, 40, 10
FRAMES = (RATE - WINDOW) // HOP + 1                       # 49
SHIFT, NOISE_CHANCE, NOISE_VOLUME = RATE // 10, 0.8, 0.1   # TensorFlow's speech_commands defaults: 100 ms, 80%, 0.1


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


def example_sources(folder: Path, seed: int = 0, all_unknown: bool = False):
    """(train, test, noise): (source, class) lists and the background-noise recordings. The
    training set is every keyword clip outside the validation and test lists, plus as many
    again "unknown" words and "silence" slices as 10% of the keyword clips each; with
    ``all_unknown``, every other word's clip is an "unknown" example, as in TensorFlow
    Datasets' split that MLPerf's reference trains on. A source is a file, or (recording,
    offset, volume) for a slice of background noise."""

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
    unknown = other_files if all_unknown else rng.sample(other_files, extra)
    train = [(path, label) for path, label in keyword_files + unknown]
    for _ in range(extra):                              # silence: a quiet slice of background noise
        source = rng.randrange(len(noise))
        train.append(((source, rng.randrange(len(noise[source]) - RATE), rng.uniform(0.0, 0.1)),
                      CLASSES.index("_silence_")))
    test = [(path, label) for label, name in enumerate(CLASSES)
            for path in sorted((folder / "test_set" / name).glob("*.wav"))]
    return train, test, noise


def waveform(source, noise) -> list[float]:
    if isinstance(source, Path):
        return one_second(read_wav(source))
    index, offset, volume = source
    return [v * volume for v in noise[index][offset : offset + RATE]]


def load_speech_commands(folder: str | Path, seed: int = 0,
                         all_unknown: bool = False) -> dict[str, list[tuple[list[float], int]]]:
    """{"train": [(mfcc, class)], "test": [...]}, computed once and cached."""

    folder = Path(folder)
    cache = folder / ("mfcc_all_unknown.pkl" if all_unknown else "mfcc.pkl")
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    train, test, noise = example_sources(folder, seed, all_unknown)
    mfcc, splits = MFCC(), {}
    for split, examples in (("train", train), ("test", test)):
        features = []
        for first in range(0, len(examples), 200):
            chunk = examples[first : first + 200]
            features += list(zip(mfcc([waveform(source, noise) for source, _ in chunk]), [label for _, label in chunk]))
        splits[split] = features
    cache.write_bytes(pickle.dumps(splits))
    return splits


def load_waveforms(folder: str | Path, seed: int = 0, all_unknown: bool = False):
    """The training clips as float32 arrays (about 2.4 GB; 16-bit samples / 32768 are exact in
    float32), in the order of load_speech_commands' training features, and the background noise."""

    train, _, noise = example_sources(Path(folder), seed, all_unknown)
    clips = []
    for source, label in train:
        silence = label == CLASSES.index("_silence_")
        samples = [0.0] * RATE if silence else waveform(source, noise)     # silence gets fresh noise each time
        clips.append((array("f", samples), label))
    return clips, noise


def augmented(samples: array, label: int, noise: list[list[float]], rng: random.Random) -> list[float]:
    """TensorFlow's speech_commands augmentation (input_data.py): shift the clip
    by up to 100 ms either way, padding with silence, then, with probability 0.8 (always for
    silence), add a random second of background noise at a random volume up to 0.1, clipped
    to [-1, 1]."""

    shift = rng.randint(-SHIFT, SHIFT)
    clip = samples.tolist()
    clip = [0.0] * shift + clip[: RATE - shift] if shift >= 0 else clip[-shift:] + [0.0] * -shift
    if label == CLASSES.index("_silence_") or rng.random() < NOISE_CHANCE:
        volume = rng.uniform(0.0, NOISE_VOLUME)
        recording = noise[rng.randrange(len(noise))]
        offset = rng.randrange(len(recording) - RATE)
        clip = [v + volume * n for v, n in zip(clip, recording[offset : offset + RATE])]
        if max(clip) > 1.0 or min(clip) < -1.0:
            clip = [min(1.0, max(-1.0, v)) for v in clip]
    return clip


def ds_cnn(seed: int = 0, dropout: bool = False) -> MobileNet:
    """MLPerf Tiny's DS-CNN: (B, 1, 49, 10) MFCCs -> 12 class scores; ``dropout`` adds
    the reference's 0.2 after the stem and 0.4 before the pooling."""

    stem_dropout, head_dropout = (0.2, 0.4) if dropout else (0.0, 0.0)
    return MobileNet(image_shape=(1, FRAMES, COEFFICIENTS), classes=len(CLASSES), stem=64, stem_kernel=(10, 4),
                     stem_stride=2, stem_padding=(5, 1), stages=((64, 1),) * 4, stem_dropout=stem_dropout,
                     head_dropout=head_dropout, seed=seed)


def standardize(train, test) -> list[tuple[float, float]]:
    """Zero mean, unit variance per MFCC coefficient, from the training clips (in place).
    Returns each coefficient's (mean, std)."""

    statistics = []
    for c in range(COEFFICIENTS):
        values = [features[t * COEFFICIENTS + c] for features, _ in train for t in range(FRAMES)]
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) or 1.0
        statistics.append((mean, std))
        for features, _ in train + test:
            for t in range(FRAMES):
                features[t * COEFFICIENTS + c] = (features[t * COEFFICIENTS + c] - mean) / std
    return statistics


def conv_kernels(model: MobileNet) -> list[Tensor]:
    """The convolution kernels, which the reference's L2 penalty covers (not the batch norm or
    the classifier)."""

    return [model.stem.kernel] + [w for block in model.blocks for w in (block.depthwise.kernel, block.pointwise.kernel)]


class NanoGradRunner:
    def __init__(self, model: MobileNet, lr: float, l2: float = 0.0) -> None:
        self.model = model
        self.optimizer = TensorAdam(model.parameters(), lr=lr)
        self.l2 = l2

    def train_step(self, features: list[float], labels: list[int], lr: float) -> float:
        self.model.train()
        self.optimizer.lr = lr
        self.optimizer.zero_grad()
        loss = softmax_cross_entropy(self.model(Tensor(features, (len(labels), 1, FRAMES, COEFFICIENTS))), labels)
        if self.l2:                                     # Keras' l2(l2): l2 * sum(w^2) on each kernel
            for weight in conv_kernels(self.model):
                loss = loss + (weight * weight).sum() * self.l2
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
    def __init__(self, model: MobileNet, lr: float, l2: float = 0.0) -> None:
        import torch
        from torch_twin import TorchTwin

        self.torch = torch
        self.twin = TorchTwin(model)
        self.optimizer = torch.optim.Adam(self.twin.parameters, lr=lr)
        kernels = {id(w) for w in conv_kernels(model)}
        self.kernels = [twin for ours, twin in zip(model.parameters(), self.twin.parameters) if id(ours) in kernels]
        self.l2 = l2

    def _inputs(self, features):
        return self.torch.tensor(features, dtype=self.torch.float64).reshape(-1, 1, FRAMES, COEFFICIENTS)

    def train_step(self, features, labels, lr: float) -> float:
        torch = self.torch
        self.twin.training = True
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(self.twin(self._inputs(features)), torch.tensor(labels))
        if self.l2:
            for weight in self.kernels:
                loss = loss + (weight * weight).sum() * self.l2
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, features, count: int) -> list[int]:
        self.twin.training = False
        with self.torch.no_grad():
            return self.twin(self._inputs(features)).argmax(1).tolist()

    def weights(self):
        return [p.detach().reshape(-1).tolist() for p in self.twin.parameters]


def accuracy(runner, examples) -> tuple[float, list[float]]:
    """The share of clips classified right, overall and for each class (of its clips)."""

    right, per_class, counts = 0, [0] * len(CLASSES), [0] * len(CLASSES)
    for first in range(0, len(examples), 500):
        chunk = examples[first : first + 500]
        predicted = runner.predict([v for f, _ in chunk for v in f], len(chunk))
        for p, (_, label) in zip(predicted, chunk):
            right += p == label
            per_class[label] += p == label
            counts[label] += 1
    return right / len(examples), [r / max(1, n) for r, n in zip(per_class, counts)]


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
    parser.add_argument("--augment", action="store_true", help="random time shifts and background noise")
    parser.add_argument("--all-unknown", action="store_true",
                        help="every other word's clip as an unknown example (TensorFlow Datasets' split)")
    parser.add_argument("--l2", type=float, default=0.0,
                        help="L2 penalty on the convolution kernels (the reference: 1e-4)")
    parser.add_argument("--dropout", action="store_true",
                        help="the reference's dropout: 0.2 after the first convolution, 0.4 before the pooling")
    args = parser.parse_args(argv)

    start = perf_counter()
    splits = load_speech_commands(args.data, args.seed, args.all_unknown)
    rng = random.Random(args.seed)
    picked = list(range(len(splits["train"])))
    rng.shuffle(picked)
    picked = picked[: args.train_limit]
    train_set = [(list(splits["train"][i][0]), splits["train"][i][1]) for i in picked]
    test_set = [(list(f), label) for f, label in splits["test"]]
    rng.shuffle(test_set)
    test_set = test_set[: args.test_limit]
    statistics = standardize(train_set, test_set)
    if args.augment:
        clips, noise = load_waveforms(args.data, args.seed, args.all_unknown)
        clips = [clips[i] for i in picked]
        mfcc = MFCC()

    def batch_features(indices: list[int], rng: random.Random) -> list[float]:
        if not args.augment:
            return [v for i in indices for v in train_set[i][0]]
        features = mfcc([augmented(*clips[i], noise, rng) for i in indices])
        return [(v - statistics[k % COEFFICIENTS][0]) / statistics[k % COEFFICIENTS][1]
                for f in features for k, v in enumerate(f)]
    print(f"{len(train_set)} training and {len(test_set)} test clips ({perf_counter() - start:.0f} s)", flush=True)
    results, runners = {}, {}
    for framework in args.frameworks:
        model = ds_cnn(seed=args.seed, dropout=args.dropout)
        if framework == "torch":
            import torch

            torch.set_num_threads(1)
            runners[framework] = runner = TorchRunner(model, args.lr, args.l2)
        else:
            runners[framework] = runner = NanoGradRunner(model, args.lr, args.l2)
        print(f"{framework} DS-CNN, {model.num_parameters()} parameters", flush=True)
        order, steps, start = list(range(len(train_set))), 0, perf_counter()
        shuffler, augmenter = random.Random(args.seed + 1), random.Random(args.seed + 2)
        for epoch in range(1, args.epochs + 1):
            lr = args.lr / 5 ** min(2, (epoch - 1) * 3 // args.epochs)       # / 5 after each third
            shuffler.shuffle(order)
            total = 0.0
            for first in range(0, len(order), args.batch_size):
                indices = order[first : first + args.batch_size]
                total += runner.train_step(batch_features(indices, augmenter), [train_set[i][1] for i in indices], lr)
                steps += 1
            print(f"  epoch {epoch:2d}: lr {lr:.0e}, train loss {total / math.ceil(len(order) / args.batch_size):.4f}, "
                  f"{perf_counter() - start:.0f} s", flush=True)
        seconds = perf_counter() - start
        overall, per_class = accuracy(runner, test_set)
        results[framework] = {"accuracy": overall, "per_class": dict(zip(CLASSES, per_class)),
                              "ms_per_step": 1000 * seconds / steps}
        print(f"{framework:8s}: test accuracy {100 * overall:.2f}% ({results[framework]['ms_per_step']:.0f} ms/step)",
              flush=True)
        print("  per class: " + ", ".join(f"{name} {100 * value:.1f}%" for name, value in zip(CLASSES, per_class)),
              flush=True)
    if len(runners) == 2:
        difference = max(abs(a - b) for x, y in zip(runners["nanograd"].weights(), runners["torch"].weights())
                         for a, b in zip(x, y))
        print(f"largest weight difference after training: {difference:.1e}"
              + (" (different dropout masks)" if args.dropout else ""))
        results["max_weight_difference"] = difference
    return results


if __name__ == "__main__":
    main()
