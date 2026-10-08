"""Super-resolution: SRCNN on the 91-image set, tested on Set5 and Set14, NanoGrad vs PyTorch.

Super-resolution makes a sharp image from a small one. SRCNN (Dong et al.,
2014) is the first deep model for it, and its protocol is still the
field's reference. A low-resolution image is first upscaled by bicubic
interpolation, which gives the right size but a blurry result. Three
convolutions then restore the detail:

    9x9 conv, 64 filters, ReLU      extract patches of the blurry image
    1x1 conv, 32 filters, ReLU      map each to a sharp-patch code (5x5 in the 9-5-5 variant)
    5x5 conv, 1 filter              reconstruct the sharp image

It works on luminance alone: the Y of YCbCr. Training pairs come from the
91-image set. Each image is shrunk 3x by bicubic interpolation and blown
back up, then cut into 33x33 patches with a stride of 14. The loss is the
mean squared error to the original. Testing follows SRCNN's MATLAB demo:
PSNR on Y in 8 bits, with 3 border pixels shaved off.

Every resize is MATLAB's ``imresize`` bicubic (a = -0.5, antialiased when
shrinking), reimplemented here, because results are only comparable with
the exact same degradation. As a check, bicubic upscaling alone scores
30.39 dB on Set5 and 27.54 dB on Set14 (x3), the numbers every paper
lists. SRCNN reports 32.39 / 29.00 dB for 9-1-5 trained on 91 images,
and 32.75 / 29.30 for 9-5-5 trained on ImageNet.

The original training ran SGD for about 10^8 patch steps. Here it is
Adam at lr 1e-4 with zero padding, as in the common PyTorch reimplementation
(yjn870/SRCNN-pytorch), for as many epochs as you give it.

    python3 sr.py --epochs 50
    python3 sr.py --bicubic-only
    .venv/bin/python sr.py --frameworks nanograd torch --epochs 1 --patch-limit 2000

Data: data/sr/{train,set5,set14}/*.bmp (see data/sr/README.md).
"""

from __future__ import annotations

import argparse
import math
import random
import struct
from pathlib import Path
from time import perf_counter

from optim import TensorAdam
from tensor import Tensor, no_grad
from tensor_nn import ReLU, Sequential, TensorConv2D

Image = list[list[float]]          # rows of values


# -- images -------------------------------------------------------------------


def read_bmp(path: str | Path) -> list[list[tuple[int, int, int]]]:
    """Rows of (R, G, B) from an uncompressed 24-bit or 8-bit palette BMP."""

    raw = Path(path).read_bytes()
    offset = struct.unpack("<I", raw[10:14])[0]
    header = struct.unpack("<I", raw[14:18])[0]
    width, height, _, bits = struct.unpack("<iiHH", raw[18:30])
    palette = []
    if bits == 8:
        colors = (offset - 14 - header) // 4
        palette = [tuple(raw[14 + header + 4 * i : 14 + header + 4 * i + 3][::-1]) for i in range(colors)]
    stride = (width * bits // 8 + 3) // 4 * 4               # rows are padded to 4 bytes
    rows = []
    for y in range(abs(height)):
        start = offset + y * stride
        if bits == 24:                                     # stored as B, G, R
            rows.append([(raw[start + 3 * x + 2], raw[start + 3 * x + 1], raw[start + 3 * x]) for x in range(width)])
        else:
            rows.append([palette[raw[start + x]] for x in range(width)])
    return rows[::-1] if height > 0 else rows               # positive height: bottom row first


def load_y(path: str | Path) -> Image:
    """An image's luminance in [0, 1], as the MATLAB evaluation code reads it: a
    one-channel (8-bit) file is used as it is, a colour one goes through rgb2ycbcr."""

    rgb = read_bmp(path)
    if struct.unpack("<H", Path(path).read_bytes()[28:30])[0] == 8:
        return [[r / 255 for r, _, _ in row] for row in rgb]
    return luminance(rgb)


def luminance(rgb: list[list[tuple[int, int, int]]]) -> Image:
    """MATLAB's rgb2ycbcr Y for 8-bit images, rounded to 8 bits, scaled to [0, 1]."""

    return [[round(16 + (65.481 * r + 128.553 * g + 24.966 * b) / 255) / 255 for r, g, b in row] for row in rgb]


def modcrop(image: Image, scale: int) -> Image:
    """Trim so both sides divide by the scale."""

    height, width = len(image) - len(image) % scale, len(image[0]) - len(image[0]) % scale
    return [row[:width] for row in image[:height]]


def cubic(x: float) -> float:
    """Keys' cubic kernel with a = -0.5, as in MATLAB's imresize."""

    x = abs(x)
    if x <= 1:
        return 1.5 * x ** 3 - 2.5 * x ** 2 + 1
    if x <= 2:
        return -0.5 * x ** 3 + 2.5 * x ** 2 - 4 * x + 2
    return 0.0


def resize_weights(size: int, scale: float) -> list[list[tuple[int, float]]]:
    """For each output position, the input positions and weights MATLAB's
    imresize mixes. When shrinking, the kernel is stretched by 1/scale, which
    averages away detail that would otherwise alias."""

    width, kernel = 4.0, cubic
    if scale < 1:
        width, kernel = 4.0 / scale, (lambda x: scale * cubic(scale * x))
    out_size = math.ceil(size * scale)
    weights = []
    for x in range(1, out_size + 1):
        u = x / scale + 0.5 * (1 - 1 / scale)               # the output pixel's centre, in input coordinates
        left = math.floor(u - width / 2)
        taps = [left + j for j in range(math.ceil(width) + 2)]
        values = [kernel(u - t) for t in taps]
        total = sum(values)
        row = {}
        for t, value in zip(taps, values):
            if value:
                # Positions outside the image mirror back in (1-based: 0 -> 1, size + 1 -> size).
                index = (t - 1) % (2 * size)
                index = index if index < size else 2 * size - 1 - index
                row[index] = row.get(index, 0.0) + value / total
        weights.append(sorted(row.items()))
    return weights


def imresize(image: Image, scale: float) -> Image:
    """Bicubic resize by ``scale``: columns of each row first, then rows (separable)."""

    across = resize_weights(len(image[0]), scale)
    down = resize_weights(len(image), scale)
    wide = [[sum(row[i] * w for i, w in taps) for taps in across] for row in image]
    return [[sum(wide[i][x] * w for i, w in taps) for x in range(len(across))] for taps in down]


def to_bytes(image: Image) -> list[list[int]]:
    """[0, 1] -> 8-bit, rounded and clipped (MATLAB's uint8(x * 255))."""

    return [[min(255, max(0, round(v * 255))) for v in row] for row in image]


def psnr(truth: Image, estimate: Image, shave: int) -> float:
    """PSNR in dB of two 8-bit images, ignoring ``shave`` pixels at each border."""

    a, b = to_bytes(truth), to_bytes(estimate)
    errors = [(a[y][x] - b[y][x]) ** 2 for y in range(shave, len(a) - shave) for x in range(shave, len(a[0]) - shave)]
    return 10 * math.log10(255 ** 2 / (sum(errors) / len(errors)))


def degrade(image: Image, scale: int) -> Image:
    """The network's input: shrink by the scale and blow back up, both bicubic."""

    return imresize(imresize(image, 1 / scale), scale)


# -- data ---------------------------------------------------------------------


def training_patches(folder: Path, scale: int, size: int = 33, stride: int = 14) -> list[tuple[list[float], list[float]]]:
    """(blurry patch, sharp patch) pairs, flattened, from every image in the folder."""

    pairs = []
    for path in sorted(folder.glob("*.bmp")):
        sharp = modcrop(load_y(path), scale)
        blurry = degrade(sharp, scale)
        for y in range(0, len(sharp) - size + 1, stride):
            for x in range(0, len(sharp[0]) - size + 1, stride):
                pairs.append(([v for row in blurry[y : y + size] for v in row[x : x + size]],
                              [v for row in sharp[y : y + size] for v in row[x : x + size]]))
    return pairs


def test_images(folder: Path, scale: int) -> list[tuple[str, Image, Image]]:
    """(name, sharp, blurry) for every image in a test folder."""

    images = []
    for path in sorted(folder.glob("*.bmp")):
        sharp = modcrop(load_y(path), scale)
        images.append((path.stem, sharp, degrade(sharp, scale)))
    return images


# -- model --------------------------------------------------------------------


def srcnn(middle: int = 1, seed: int = 0) -> Sequential:
    """9-1-5 (middle=1) or 9-5-5 SRCNN, zero-padded so the output keeps the input's size."""

    return Sequential(
        TensorConv2D((64, 1, 9, 9), padding=4, seed=seed),
        ReLU(),
        TensorConv2D((32, 64, middle, middle), padding=middle // 2, seed=seed + 1),
        ReLU(),
        TensorConv2D((1, 32, 5, 5), padding=2, seed=seed + 2),
    )


class NanoGradRunner:
    def __init__(self, model: Sequential, lr: float) -> None:
        self.model = model
        self.optimizer = TensorAdam(model.parameters(), lr=lr)

    def train_step(self, inputs: list[float], targets: list[float], shape: tuple[int, ...]) -> float:
        self.optimizer.zero_grad()
        error = self.model(Tensor(inputs, shape)) - Tensor(targets, shape)
        loss = (error * error).mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, image: Image) -> Image:
        height, width = len(image), len(image[0])
        with no_grad():
            out = self.model(Tensor([v for row in image for v in row], (1, 1, height, width))).data
        return [list(out[y * width : (y + 1) * width]) for y in range(height)]

    def weights(self) -> list[list[float]]:
        return [list(p.data) for p in self.model.parameters()]


class TorchRunner:
    def __init__(self, model: Sequential, lr: float) -> None:
        import torch
        from torch_twin import TorchTwin

        self.torch = torch
        self.twin = TorchTwin(model)
        self.optimizer = torch.optim.Adam(self.twin.parameters, lr=lr)

    def train_step(self, inputs, targets, shape) -> float:
        torch = self.torch
        self.optimizer.zero_grad()
        out = self.twin(torch.tensor(inputs, dtype=torch.float64).reshape(shape))
        loss = ((out - torch.tensor(targets, dtype=torch.float64).reshape(shape)) ** 2).mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, image: Image) -> Image:
        torch = self.torch
        with torch.no_grad():
            out = self.twin(torch.tensor(image, dtype=torch.float64)[None, None])
        return out[0, 0].tolist()

    def weights(self) -> list[list[float]]:
        return [p.detach().reshape(-1).tolist() for p in self.twin.parameters]


def evaluate(predict, images, scale: int) -> float:
    """Mean PSNR over a test set (dB); ``predict`` maps the blurry image to the estimate."""

    return sum(psnr(sharp, predict(blurry), scale) for _, sharp, blurry in images) / len(images)


def train(runner, patches, test_sets, args) -> dict:
    rng = random.Random(args.seed)
    order = list(range(len(patches)))
    size = int(math.sqrt(len(patches[0][0])))
    history, steps, training, start = [], 0, 0.0, perf_counter()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(order)
        total, batches, began = 0.0, 0, perf_counter()
        for first in range(0, len(order), args.batch_size):
            batch = [patches[i] for i in order[first : first + args.batch_size]]
            shape = (len(batch), 1, size, size)
            total += runner.train_step([v for x, _ in batch for v in x], [v for _, y in batch for v in y], shape)
            batches += 1
        steps += batches
        training += perf_counter() - began
        scores = {name: evaluate(runner.predict, images, args.scale) for name, images in test_sets.items()}
        history.append((total / batches, scores))
        print(f"  epoch {epoch:3d}: loss {total / batches:.6f}, "
              + ", ".join(f"{name} {score:.2f} dB" for name, score in scores.items())
              + f", {perf_counter() - start:.0f} s", flush=True)
    return {"history": history, "psnr": history[-1][1], "ms_per_step": 1000 * training / max(1, steps)}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="data/sr")
    parser.add_argument("--frameworks", nargs="+", choices=("nanograd", "torch"), default=["nanograd"])
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--middle", type=int, choices=(1, 5), default=1, help="1: SRCNN 9-1-5, 5: 9-5-5")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch-limit", type=int, help="use only the first N training patches")
    parser.add_argument("--test-sets", nargs="+", default=["set5", "set14"])
    parser.add_argument("--bicubic-only", action="store_true", help="just score bicubic upscaling")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    data = Path(args.data)
    test_sets = {name: test_images(data / name, args.scale) for name in args.test_sets}
    results = {"bicubic": {name: evaluate(lambda blurry: blurry, images, args.scale)
                           for name, images in test_sets.items()}}
    print("bicubic: " + ", ".join(f"{name} {score:.2f} dB" for name, score in results["bicubic"].items()), flush=True)
    if args.bicubic_only:
        return results
    patches = training_patches(data / "train", args.scale)[: args.patch_limit]
    print(f"{len(patches)} training patches", flush=True)
    runners = {}
    for framework in args.frameworks:
        model = srcnn(args.middle, seed=args.seed)
        if framework == "torch":
            import torch

            torch.set_num_threads(1)
            runners[framework] = TorchRunner(model, args.lr)
        else:
            runners[framework] = NanoGradRunner(model, args.lr)
        print(f"{framework} SRCNN 9-{args.middle}-5", flush=True)
        results[framework] = result = train(runners[framework], patches, test_sets, args)
        print(f"{framework:8s} 9-{args.middle}-5: " + ", ".join(f"{n} {s:.2f} dB" for n, s in result["psnr"].items())
              + f", {result['ms_per_step']:.0f} ms/step", flush=True)
    if len(runners) == 2:
        difference = max(abs(a - b) for x, y in zip(runners["nanograd"].weights(), runners["torch"].weights())
                         for a, b in zip(x, y))
        print(f"largest weight difference after training: {difference:.1e}")
        results["max_weight_difference"] = difference
    return results


if __name__ == "__main__":
    main()
