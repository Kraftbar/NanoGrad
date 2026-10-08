"""Few-shot learning: prototypical networks on Omniglot, NanoGrad vs PyTorch.

Most classifiers see thousands of examples per class. Few-shot learning
asks for a classifier of N new classes from K examples of each, often one
(N-way K-shot). Omniglot (Lake et al., 2015) is the standard test: 1,623
handwritten characters from 50 alphabets, 20 drawings of each. Following
Vinyals et al. (2016), 1,200 characters (with rotations by 90, 180 and 270
degrees as extra classes: 4,800) are for training, and the 423 others
(1,692 with rotations) for testing. The test characters are never seen
during training.

A prototypical network (Snell et al., 2017) learns an embedding in which
classes cluster. To classify, it averages the K support embeddings of
each class into a prototype c_n, and scores a query x by distance:

    p(class n | x) = softmax_n(-||f(x) - c_n||^2)

f is four blocks of 3x3 convolution (64 filters), batch norm, ReLU and 2x2
max pooling; a 28x28 image comes out as 64 numbers. Training mimics the
test. Each step is one "episode": pick 60 random training classes, take
K support and 5 query images of each, and minimize the cross entropy of
the queries. Adam runs at lr 1e-3, halved every 2,000 episodes.

The paper reports test accuracy (1,000 episodes, 5-way and 20-way, 1 and
5 shots): 98.8% / 99.7% for 5-way, 96.0% / 98.9% for 20-way. Their 5-shot
numbers come from a model trained on 5-shot episodes; here one model,
trained with ``--train-shot`` (1 by default), is tested at both.

    python3 fewshot.py --episodes 5000
    .venv/bin/python fewshot.py --frameworks nanograd torch --episodes 20 --train-way 20

Data: data/omniglot/ (see data/omniglot/README.md).
"""

from __future__ import annotations

import argparse
import pickle
import random
import zipfile
from pathlib import Path
from time import perf_counter

from images import read_png
from losses import softmax_cross_entropy
from optim import TensorAdam
from tensor import Tensor, no_grad
from tensor_nn import MaxPool2d, ReLU, Sequential, TensorBatchNorm2d, TensorConv2D

SIZE = 28


# -- data ---------------------------------------------------------------------


def rotate(image: list[list[int]], degrees: int) -> list[list[int]]:
    """Rotate counter-clockwise by a multiple of 90 degrees."""

    for _ in range(degrees // 90 % 4):
        image = [list(row) for row in zip(*image)][::-1]
    return image


def shrink(image: list[list[int]], size: int = SIZE) -> bytes:
    """Nearest-neighbour resize to size x size (what PIL does for a 1-bit image),
    flattened, with ink as 1 and paper as 0."""

    height, width = len(image), len(image[0])
    return bytes(1 - image[int((y + 0.5) * height / size)][int((x + 0.5) * width / size)] // 255
                 for y in range(size) for x in range(size))


def load_omniglot(folder: str | Path) -> dict[str, dict[str, list[bytes]]]:
    """{split: {"Alphabet/characterNN/rotDDD": [20 images, 28 x 28 bytes of 0/1]}}, for
    the Vinyals splits, decoded once and cached in the folder."""

    folder = Path(folder)
    cache = folder / f"omniglot{SIZE}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    drawings: dict[str, list[list[list[int]]]] = {}
    for archive in ("images_background.zip", "images_evaluation.zip"):
        with zipfile.ZipFile(folder / archive) as zipped:
            for name in sorted(zipped.namelist()):
                if name.endswith(".png"):
                    _, alphabet, character, _ = name.split("/")
                    drawings.setdefault(f"{alphabet}/{character}", []).append(read_png(zipped.read(name)))
    splits = {}
    for split in ("train", "val", "test"):
        classes = {}
        for line in (folder / f"vinyals_{split}.txt").read_text().split():
            character, rotation = line.rsplit("/", 1)
            classes[line] = [shrink(rotate(image, int(rotation[3:]))) for image in drawings[character]]
        splits[split] = classes
    cache.write_bytes(pickle.dumps(splits))
    return splits


# -- model --------------------------------------------------------------------


def encoder(seed: int = 0, channels: int = 64) -> Sequential:
    """Four conv-BN-ReLU-pool blocks: (B, 1, 28, 28) -> (B, 64, 1, 1)."""

    layers, inputs = [], 1
    for block in range(4):
        # No conv bias: the batch norm right after subtracts any constant anyway.
        layers += [TensorConv2D((channels, inputs, 3, 3), padding=1, use_bias=False, seed=seed + block),
                   TensorBatchNorm2d(channels), ReLU(), MaxPool2d(2)]
        inputs = channels
    return Sequential(*layers)


def episode_batch(classes: dict[str, list[list[int]]], way: int, shot: int, query: int,
                  rng: random.Random) -> tuple[list[float], list[float], int]:
    """Support images (way * shot, class by class), then query images (way * query)."""

    support, queries = [], []
    for name in rng.sample(sorted(classes), way):
        picks = rng.sample(classes[name], shot + query)
        support += [v for image in picks[:shot] for v in image]
        queries += [v for image in picks[shot:] for v in image]
    return support, queries, way


class NanoGradRunner:
    def __init__(self, model: Sequential, lr: float) -> None:
        self.model = model
        self.optimizer = TensorAdam(model.parameters(), lr=lr)

    def _scores(self, support, queries, way: int) -> Tensor:
        """-squared distance from every query to every class prototype: (queries, way)."""

        images = Tensor(support + queries, ((len(support) + len(queries)) // SIZE ** 2, 1, SIZE, SIZE))
        embedded = self.model(images)
        dim = embedded.numel // images.shape[0]
        embedded = embedded.reshape((images.shape[0], dim))
        shots = len(support) // SIZE ** 2 // way
        prototypes = embedded[: way * shots].reshape((way, shots, dim)).mean(axis=1)            # (way, dim)
        q = embedded[way * shots :]                                                            # (Q, dim)
        # ||q - c||^2 = ||q||^2 - 2 q . c + ||c||^2
        return (q @ prototypes.T) * 2.0 - (q * q).sum(axis=1).reshape((q.shape[0], 1)) \
            - (prototypes * prototypes).sum(axis=1).reshape((1, way))

    def train_step(self, support, queries, way: int, lr: float) -> tuple[float, float]:
        self.model.train()
        self.optimizer.lr = lr
        self.optimizer.zero_grad()
        scores = self._scores(support, queries, way)
        per_class = scores.shape[0] // way
        labels = [n for n in range(way) for _ in range(per_class)]
        loss = softmax_cross_entropy(scores, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item(), accuracy(scores.data, labels, way)

    def embed(self, images: list[float]) -> list[list[float]]:
        self.model.eval()
        count = len(images) // SIZE ** 2
        with no_grad():
            out = self.model(Tensor(images, (count, 1, SIZE, SIZE))).data
        dim = len(out) // count
        return [list(out[i * dim : (i + 1) * dim]) for i in range(count)]

    def weights(self) -> list[list[float]]:
        return [list(p.data) for p in self.model.parameters()]


class TorchRunner:
    def __init__(self, model: Sequential, lr: float) -> None:
        import torch
        from torch_twin import TorchTwin

        self.torch = torch
        self.twin = TorchTwin(model)
        self.optimizer = torch.optim.Adam(self.twin.parameters, lr=lr)

    def train_step(self, support, queries, way: int, lr: float) -> tuple[float, float]:
        torch = self.torch
        self.twin.training = True
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad()
        images = torch.tensor(support + queries, dtype=torch.float64).reshape(-1, 1, SIZE, SIZE)
        embedded = self.twin(images).flatten(1)
        shots = len(support) // SIZE ** 2 // way
        prototypes = embedded[: way * shots].reshape(way, shots, -1).mean(dim=1)
        q = embedded[way * shots :]
        scores = -torch.cdist(q, prototypes) ** 2
        labels = torch.arange(way).repeat_interleave(q.shape[0] // way)
        loss = torch.nn.functional.cross_entropy(scores, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item(), accuracy(scores.detach().reshape(-1).tolist(), labels.tolist(), way)

    def embed(self, images: list[float]) -> list[list[float]]:
        torch = self.torch
        self.twin.training = False
        with torch.no_grad():
            return self.twin(torch.tensor(images, dtype=torch.float64).reshape(-1, 1, SIZE, SIZE)).flatten(1).tolist()

    def weights(self) -> list[list[float]]:
        return [p.detach().reshape(-1).tolist() for p in self.twin.parameters]


def accuracy(scores, labels: list[int], way: int) -> float:
    right = sum(max(range(way), key=lambda n: scores[i * way + n]) == label for i, label in enumerate(labels))
    return right / len(labels)


def evaluate(runner, classes, *, way: int, shot: int, query: int, episodes: int, seed: int,
             embeddings: dict[str, list[list[float]]] | None = None) -> float:
    """Mean query accuracy over random test episodes. With batch norm in
    evaluation mode an image's embedding does not depend on the rest of the
    batch, so every test image is embedded once and episodes reuse them."""

    if embeddings is None:
        embeddings = {}
        for name, images in classes.items():
            embeddings[name] = runner.embed([v for image in images for v in image])
    rng, total = random.Random(seed), 0.0
    for _ in range(episodes):
        prototypes, queries = [], []
        for label, name in enumerate(rng.sample(sorted(classes), way)):
            picks = rng.sample(range(len(embeddings[name])), shot + query)
            vectors = [embeddings[name][i] for i in picks]
            prototypes.append([sum(column) / shot for column in zip(*vectors[:shot])])
            queries += [(vector, label) for vector in vectors[shot:]]
        right = 0
        for vector, label in queries:
            distances = [sum((a - b) ** 2 for a, b in zip(vector, prototype)) for prototype in prototypes]
            right += min(range(way), key=distances.__getitem__) == label
        total += right / len(queries)
    return total / episodes


def embed_all(runner, classes) -> dict[str, list[list[float]]]:
    names = sorted(classes)
    embeddings = {}
    for first in range(0, len(names), 25):              # 500 images per forward pass
        chunk = names[first : first + 25]
        vectors = runner.embed([v for name in chunk for image in classes[name] for v in image])
        for index, name in enumerate(chunk):
            embeddings[name] = vectors[index * 20 : (index + 1) * 20]
    return embeddings


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="data/omniglot")
    parser.add_argument("--frameworks", nargs="+", choices=("nanograd", "torch"), default=["nanograd"])
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--train-way", type=int, default=60)
    parser.add_argument("--train-shot", type=int, default=1)
    parser.add_argument("--train-query", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--halve-every", type=int, default=2000)
    parser.add_argument("--test-episodes", type=int, default=1000)
    parser.add_argument("--test-query", type=int, default=15)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    splits = load_omniglot(args.data)
    train_classes = {**splits["train"], **splits["val"]}           # Snell et al. train on train + val
    print(f"{len(train_classes)} training classes, {len(splits['test'])} test classes", flush=True)
    results, runners = {}, {}
    for framework in args.frameworks:
        model = encoder(seed=args.seed)
        if framework == "torch":
            import torch

            torch.set_num_threads(1)
            runners[framework] = runner = TorchRunner(model, args.lr)
        else:
            runners[framework] = runner = NanoGradRunner(model, args.lr)
        print(framework, flush=True)
        rng, start, recent = random.Random(args.seed), perf_counter(), []
        training = []   # seconds per train_step, without sampling the episode
        for episode in range(1, args.episodes + 1):
            lr = args.lr * 0.5 ** ((episode - 1) // args.halve_every)
            batch = episode_batch(train_classes, args.train_way, args.train_shot, args.train_query, rng)
            began = perf_counter()
            recent.append(runner.train_step(*batch, lr))
            training.append(perf_counter() - began)
            if episode % args.log_every == 0 or episode == args.episodes:
                print(f"  episode {episode:5d}: loss {sum(r[0] for r in recent) / len(recent):.4f}, "
                      f"train accuracy {100 * sum(r[1] for r in recent) / len(recent):.1f}%, "
                      f"{perf_counter() - start:.0f} s", flush=True)
                recent = []
        timed = training[1:] or training   # the first step also builds the convolution index tables
        embeddings = embed_all(runner, splits["test"])
        scores = {}
        for way in (5, 20):
            for shot in (1, 5):
                scores[f"{way}-way {shot}-shot"] = evaluate(runner, splits["test"], way=way, shot=shot,
                                                            query=args.test_query, episodes=args.test_episodes,
                                                            seed=args.seed + 1, embeddings=embeddings)
        results[framework] = {"accuracy": scores, "ms_per_episode": 1000 * sum(timed) / len(timed)}
        print(f"{framework:8s}: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in scores.items())
              + f" ({results[framework]['ms_per_episode']:.0f} ms per training episode)", flush=True)
    if len(runners) == 2:
        difference = max(abs(a - b) for x, y in zip(runners["nanograd"].weights(), runners["torch"].weights())
                         for a, b in zip(x, y))
        print(f"largest weight difference after training: {difference:.1e}")
        results["max_weight_difference"] = difference
    return results


if __name__ == "__main__":
    main()
