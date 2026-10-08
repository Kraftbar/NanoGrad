"""Semantic segmentation: a U-Net on CamVid street scenes, NanoGrad vs PyTorch.

Classification gives one label per image. Segmentation gives one per
pixel: road, car, pedestrian, sky... CamVid (Brostow et al., 2008) is
the classic small benchmark: 701 frames of driving video, each pixel
labelled with one of 11 classes (or "void", ignored). SegNet's split is
367 images to train on, 101 to validate and 233 to test.

A U-Net (Ronneberger et al., 2015) is an encoder-decoder with shortcuts:

    encoder   conv blocks, each followed by 2x2 max pooling: the image shrinks,
              the features grow more abstract (what is where, coarsely)
    decoder   transposed convolutions grow it back; at each scale the encoder's
              features from that scale are concatenated in, restoring the detail
              (where exactly the edges are) that pooling threw away
    head      a 1x1 convolution: 11 class scores at every pixel

The loss is the cross entropy of every non-void pixel. Each conv block is
two 3x3 convolutions with batch norm and ReLU; widths 16-32-64, 128 at the
bottom. To fit a CPU, images are halved to 180 x 240 (2 x 2 averages;
labels take the top-left pixel), and the centre 176 x 240 is used so three
poolings divide evenly. Adam at lr 1e-3, batch 4, random horizontal flips.

Reported: global pixel accuracy, class-average accuracy, and mean
intersection-over-union over the 11 classes, on the 233 test images.

    python3 segment.py --epochs 30
    .venv/bin/python segment.py --frameworks nanograd torch --epochs 1 --train-limit 8 --test-limit 4

Data: data/camvid/ (see data/camvid/README.md).
"""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path
from time import perf_counter

import kernels
from images import read_png
from losses import softmax_cross_entropy
from optim import TensorAdam
from tensor import Tensor, concat, gather_rows, no_grad
from tensor_nn import (
    MaxPool2d, ReLU, Sequential, TensorBatchNorm2d, TensorConv2D, TensorConvTranspose2D, TensorModule,
)

CLASSES = ("sky", "building", "pole", "road", "pavement", "tree", "sign", "fence", "car", "pedestrian", "bicyclist")
VOID = 11
HEIGHT, WIDTH = 176, 240          # the centre of the half-size frame (180 x 240)


# -- data ---------------------------------------------------------------------


def half(rows: list[list], *, average: bool) -> list[list]:
    """Halve both sides: 2x2 averages of RGB pixels, or the top-left of each 2x2 label block."""

    if not average:
        return [row[::2] for row in rows[::2]]
    out = []
    for y in range(0, len(rows) - 1, 2):
        top, bottom = rows[y], rows[y + 1]
        out.append([tuple((top[x][c] + top[x + 1][c] + bottom[x][c] + bottom[x + 1][c]) / 4 for c in range(3))
                    for x in range(0, len(top) - 1, 2)])
    return out


def load_camvid(folder: str | Path) -> dict[str, list[tuple[bytes, bytes]]]:
    """{split: [(RGB bytes, label bytes)]} at HEIGHT x WIDTH, decoded once and cached."""

    folder = Path(folder)
    cache = folder / f"camvid_{HEIGHT}x{WIDTH}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    splits = {}
    for split in ("train", "val", "test"):
        pairs = []
        for line in (folder / f"{split}.txt").read_text().split("\n"):
            if not line.strip():
                continue
            image_path, label_path = (Path(p).relative_to("/SegNet/CamVid") for p in line.split())
            rgb = half(read_png((folder / image_path).read_bytes()), average=True)
            labels = half(read_png((folder / label_path).read_bytes()), average=False)
            top = (len(rgb) - HEIGHT) // 2
            rgb, labels = rgb[top : top + HEIGHT], labels[top : top + HEIGHT]
            pairs.append((bytes(round(pixel[c]) for c in range(3) for row in rgb for pixel in row[:WIDTH]),
                          bytes(v for row in labels for v in row[:WIDTH])))
        splits[split] = pairs
    cache.write_bytes(pickle.dumps(splits))
    return splits


def channel_stats(pairs) -> list[tuple[float, float]]:
    """Mean and standard deviation of each colour channel over the training images."""

    plane = HEIGHT * WIDTH
    stats = []
    for c in range(3):
        total = squares = 0.0
        for rgb, _ in pairs:
            values = rgb[c * plane : (c + 1) * plane]
            total += sum(values)
            squares += sum(v * v for v in values)
        count = plane * len(pairs)
        mean = total / count
        stats.append((mean, (squares / count - mean * mean) ** 0.5))
    return stats


def batch_arrays(pairs, stats, flips: list[bool]) -> tuple[list[float], list[int]]:
    """Normalized (B, 3, H, W) pixels and (B * H * W) labels, mirrored where flips[b]."""

    plane = HEIGHT * WIDTH
    pixels, labels = [], []
    for (rgb, label), flip in zip(pairs, flips):
        for c, (mean, std) in enumerate(stats):
            channel = rgb[c * plane : (c + 1) * plane]
            for y in range(HEIGHT):
                row = channel[y * WIDTH : (y + 1) * WIDTH]
                pixels += [(v - mean) / std for v in (row[::-1] if flip else row)]
        for y in range(HEIGHT):
            row = label[y * WIDTH : (y + 1) * WIDTH]
            labels += row[::-1] if flip else row
    return pixels, labels


# -- model --------------------------------------------------------------------


def conv_block(inputs: int, outputs: int, seed: int) -> Sequential:
    """Two 3x3 convolutions, each with batch norm and ReLU."""

    return Sequential(
        TensorConv2D((outputs, inputs, 3, 3), padding=1, use_bias=False, seed=seed), TensorBatchNorm2d(outputs), ReLU(),
        TensorConv2D((outputs, outputs, 3, 3), padding=1, use_bias=False, seed=seed + 1), TensorBatchNorm2d(outputs),
        ReLU(),
    )


class UNet(TensorModule):
    def __init__(self, classes: int = len(CLASSES), widths: tuple[int, ...] = (16, 32, 64, 128), seed: int = 0) -> None:
        self.down = [conv_block(3 if i == 0 else widths[i - 1], w, seed + 10 * i) for i, w in enumerate(widths[:-1])]
        self.pool = MaxPool2d(2)
        self.middle = conv_block(widths[-2], widths[-1], seed + 100)
        self.ups, self.up_blocks = [], []
        for i in range(len(widths) - 1, 0, -1):          # 128 -> 64, 64 -> 32, 32 -> 16
            self.ups.append(TensorConvTranspose2D(widths[i], widths[i - 1], 2, stride=2, seed=seed + 200 + i))
            self.up_blocks.append(conv_block(2 * widths[i - 1], widths[i - 1], seed + 300 + 10 * i))
        self.head = TensorConv2D((classes, widths[0], 1, 1), seed=seed + 400)

    def __call__(self, x: Tensor) -> Tensor:
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.middle(x)
        for up, block, skip in zip(self.ups, self.up_blocks, reversed(skips)):
            x = block(concat([skip, up(x)], axis=1))     # the encoder's detail, then the decoder's context
        return self.head(x)

    def parameters(self) -> list[Tensor]:
        parts = [*self.down, self.middle, *[m for pair in zip(self.ups, self.up_blocks) for m in pair], self.head]
        return [p for part in parts for p in part.parameters()]


def pixel_loss(scores: Tensor, labels: list[int]) -> Tensor:
    """Mean cross entropy over the non-void pixels; scores are (B, C, H, W)."""

    batch, classes, height, width = scores.shape
    per_pixel = scores.permute((0, 2, 3, 1)).reshape((batch * height * width, classes))
    keep = [i for i, label in enumerate(labels) if label != VOID]
    return softmax_cross_entropy(gather_rows(per_pixel, keep), [labels[i] for i in keep])


class NanoGradRunner:
    def __init__(self, model: UNet, lr: float) -> None:
        self.model = model
        self.optimizer = TensorAdam(model.parameters(), lr=lr)

    def train_step(self, pixels: list[float], labels: list[int], batch: int) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        loss = pixel_loss(self.model(Tensor(pixels, (batch, 3, HEIGHT, WIDTH))), labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, pixels: list[float], batch: int) -> list[int]:
        self.model.eval()
        with no_grad():
            scores = self.model(Tensor(pixels, (batch, 3, HEIGHT, WIDTH))).data
        return argmax_classes(scores, batch)

    def weights(self) -> list[list[float]]:
        return [list(p.data) for p in self.model.parameters()]


class TorchRunner:
    def __init__(self, model: UNet, lr: float) -> None:
        import torch
        from torch_twin import TorchTwin

        self.torch = torch
        self.twin = TorchTwin(model)
        self.optimizer = torch.optim.Adam(self.twin.parameters, lr=lr)

    def train_step(self, pixels, labels, batch: int) -> float:
        torch = self.torch
        self.twin.training = True
        self.optimizer.zero_grad()
        scores = self.twin(torch.tensor(pixels, dtype=torch.float64).reshape(batch, 3, HEIGHT, WIDTH))
        loss = torch.nn.functional.cross_entropy(scores, torch.tensor(labels).reshape(batch, HEIGHT, WIDTH),
                                                 ignore_index=VOID)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, pixels, batch: int) -> list[int]:
        torch = self.torch
        self.twin.training = False
        with torch.no_grad():
            scores = self.twin(torch.tensor(pixels, dtype=torch.float64).reshape(batch, 3, HEIGHT, WIDTH))
        return scores.argmax(1).reshape(-1).tolist()

    def weights(self):
        return [p.detach().reshape(-1).tolist() for p in self.twin.parameters]


def argmax_classes(scores, batch: int) -> list[int]:
    """The best class at every pixel, from (B, C, H, W) scores, in (B, H, W) order."""

    plane, classes = HEIGHT * WIDTH, len(scores) // (batch * HEIGHT * WIDTH)
    out = []
    for b in range(batch):
        base = b * classes * plane
        best = list(scores[base : base + plane])
        winner = [0] * plane
        for c in range(1, classes):
            channel = scores[base + c * plane : base + (c + 1) * plane]
            for i, value in enumerate(channel):
                if value > best[i]:
                    best[i], winner[i] = value, c
        out += winner
    return out


def evaluate(runner, pairs, stats, batch: int = 4) -> dict:
    """Global accuracy, class-average accuracy and mean IoU over the non-void pixels."""

    classes = len(CLASSES)
    confusion = [[0] * classes for _ in range(classes)]       # confusion[truth][predicted]
    for first in range(0, len(pairs), batch):
        chunk = pairs[first : first + batch]
        pixels, labels = batch_arrays(chunk, stats, [False] * len(chunk))
        for truth, guess in zip(labels, runner.predict(pixels, len(chunk))):
            if truth != VOID:
                confusion[truth][guess] += 1
    total = sum(map(sum, confusion))
    right = [confusion[c][c] for c in range(classes)]
    present = [c for c in range(classes) if sum(confusion[c])]
    class_accuracy = sum(right[c] / sum(confusion[c]) for c in present) / len(present)
    ious = [right[c] / (sum(confusion[c]) + sum(row[c] for row in confusion) - right[c]) for c in present]
    return {"global": sum(right) / total, "class_average": class_accuracy, "mean_iou": sum(ious) / len(ious),
            "iou": {CLASSES[c]: iou for c, iou in zip(present, ious)}}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="data/camvid")
    parser.add_argument("--frameworks", nargs="+", choices=("nanograd", "torch"), default=["nanograd"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--dtype", choices=("float64", "float32"), help="NanoGrad's precision (PyTorch stays float64)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.dtype:
        kernels.set_default_dtype(args.dtype)

    start = perf_counter()
    splits = load_camvid(args.data)
    train_pairs, test_pairs = splits["train"][: args.train_limit], splits["test"][: args.test_limit]
    stats = channel_stats(train_pairs)
    print(f"{len(train_pairs)} training and {len(test_pairs)} test images at {HEIGHT} x {WIDTH} "
          f"({perf_counter() - start:.0f} s to load)", flush=True)
    results, runners = {}, {}
    for framework in args.frameworks:
        model = UNet(seed=args.seed)
        if framework == "torch":
            import torch

            torch.set_num_threads(1)
            runners[framework] = runner = TorchRunner(model, args.lr)
        else:
            runners[framework] = runner = NanoGradRunner(model, args.lr)
        print(framework, flush=True)
        rng, order, steps, start = random.Random(args.seed), list(range(len(train_pairs))), 0, perf_counter()
        for epoch in range(1, args.epochs + 1):
            rng.shuffle(order)
            total = 0.0
            for first in range(0, len(order), args.batch_size):
                chunk = [train_pairs[i] for i in order[first : first + args.batch_size]]
                pixels, labels = batch_arrays(chunk, stats, [rng.random() < 0.5 for _ in chunk])
                total += runner.train_step(pixels, labels, len(chunk))
                steps += 1
            print(f"  epoch {epoch:2d}: loss {total / -(-len(order) // args.batch_size):.4f}, "
                  f"{perf_counter() - start:.0f} s", flush=True)
        seconds = perf_counter() - start
        results[framework] = scores = evaluate(runner, test_pairs, stats)
        scores["ms_per_step"] = 1000 * seconds / steps
        print(f"{framework:8s}: global accuracy {100 * scores['global']:.1f}%, class average "
              f"{100 * scores['class_average']:.1f}%, mean IoU {100 * scores['mean_iou']:.1f}% "
              f"({scores['ms_per_step']:.0f} ms/step)", flush=True)
        print("  IoU: " + ", ".join(f"{name} {100 * v:.0f}" for name, v in scores["iou"].items()), flush=True)
    if len(runners) == 2:
        difference = max(abs(a - b) for x, y in zip(runners["nanograd"].weights(), runners["torch"].weights())
                         for a, b in zip(x, y))
        print(f"largest weight difference after training: {difference:.1e}")
        results["max_weight_difference"] = difference
    return results


if __name__ == "__main__":
    main()
