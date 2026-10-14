"""How good are generated 28 x 28 images? A Frechet distance on a classifier's features.

FID (Heusel et al., 2017) compares two sets of images through the features
a trained classifier sees in them: fit a Gaussian (mean mu, covariance S)
to each set's features, and report the Frechet distance between the two

    |mu_1 - mu_2|^2 + tr(S_1 + S_2 - 2 (S_1^1/2 S_2 S_1^1/2)^1/2)

which is 0 only when the means and covariances match. The usual feature
extractor is Inception-v3 on 299 x 299 colour images; for MNIST-sized
images a small CNN trained on the same dataset is the common stand-in
(as in many diffusion and GAN papers' MNIST experiments), so the numbers
are comparable within this file's setup, not with Inception FIDs.

The classifier here is two 3 x 3 convolutions (32 and 64 channels, ReLU,
2 x 2 max pooling), a 128-wide ReLU layer whose outputs are the features,
and a 10-way output, trained for two epochs of Adam: about 99% test
accuracy on MNIST and 91% on Fashion-MNIST. It also labels the samples,
so the report includes how often each class was generated.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from linalg import symmetric_eigen
from losses import softmax_cross_entropy
from mnist_compare import DATASETS, epoch_batches, load_split
from optim import TensorAdam
from tensor import Tensor, matmul, no_grad
from tensor_nn import TensorConv2D, TensorLinear, TensorModule

SIGNED = [value / 127.5 - 1.0 for value in range(256)]      # byte -> value in [-1, 1], as the diffusion model sees it


class FeatureNet(TensorModule):
    def __init__(self, seed: int = 0) -> None:
        self.conv1 = TensorConv2D((32, 1, 3, 3), padding=1, seed=seed)
        self.conv2 = TensorConv2D((64, 32, 3, 3), padding=1, seed=seed + 1)
        self.hidden = TensorLinear(64 * 7 * 7, 128, seed=seed + 2)
        self.out = TensorLinear(128, 10, seed=seed + 3)

    def features(self, images: Tensor) -> Tensor:
        x = self.conv1(images).relu().max_pool2d((2, 2))
        x = self.conv2(x).relu().max_pool2d((2, 2))
        return self.hidden(x.flatten(start_axis=1)).relu()

    def __call__(self, images: Tensor) -> Tensor:
        return self.out(self.features(images))

    def parameters(self) -> list[Tensor]:
        return [p for layer in (self.conv1, self.conv2, self.hidden, self.out) for p in layer.parameters()]


def train_feature_net(dataset: str, *, epochs: int = 2, seed: int = 0, batch_size: int = 64) -> FeatureNet:
    images, labels = load_split(dataset, Path(DATASETS[dataset][0]), "train", None)
    model = FeatureNet(seed)
    optimizer = TensorAdam(model.parameters(), lr=1e-3)
    rng = random.Random(seed)
    for _ in range(epochs):
        for batch in epoch_batches(len(images), batch_size, rng):
            pixels = Tensor([SIGNED[b] for i in batch for b in images[i]], (len(batch), 1, 28, 28))
            optimizer.zero_grad()
            softmax_cross_entropy(model(pixels), [labels[i] for i in batch]).backward()
            optimizer.step()
    return model


def run_in_batches(model: FeatureNet, values: list[float], batch_size: int = 500) -> tuple[list[list[float]], list[int]]:
    """(128 features per image, predicted class per image) for flat [-1, 1] images."""

    features, classes = [], []
    count = len(values) // 784
    with no_grad():
        for start in range(0, count, batch_size):
            end = min(count, start + batch_size)
            hidden = model.features(Tensor(values[start * 784 : end * 784], (end - start, 1, 28, 28)))
            scores = model.out(hidden).data
            width = hidden.shape[1]
            for row in range(end - start):
                features.append(list(hidden.data[row * width : (row + 1) * width]))
                lane = scores[row * 10 : (row + 1) * 10]
                classes.append(max(range(10), key=lane.__getitem__))
    return features, classes


def gaussian(features: list[list[float]]) -> tuple[list[float], list[list[float]]]:
    """Mean and (unbiased) covariance of feature rows."""

    count, width = len(features), len(features[0])
    mean = [sum(row[j] for row in features) / count for j in range(width)]
    centred = Tensor([value - mean[j] for row in features for j, value in enumerate(row)], (count, width))
    covariance = matmul(centred.T, centred).data
    return mean, [[covariance[i * width + j] / (count - 1) for j in range(width)] for i in range(width)]


def _matrix_sqrt(matrix: list[list[float]]) -> list[list[float]]:
    """Symmetric square root through the eigen decomposition (negative rounding clipped to 0)."""

    values, vectors = symmetric_eigen(matrix, tolerance=1e-12)
    size = len(matrix)
    roots = [math.sqrt(max(0.0, value)) for value in values]
    return [[sum(roots[k] * vectors[k][i] * vectors[k][j] for k in range(size)) for j in range(size)]
            for i in range(size)]


def frechet_distance(first: tuple[list[float], list[list[float]]],
                     second: tuple[list[float], list[list[float]]]) -> float:
    (mu1, s1), (mu2, s2) = first, second
    size = len(mu1)
    root = _matrix_sqrt(s1)
    flat = lambda m: Tensor([v for row in m for v in row], (size, size))      # noqa: E731
    middle = matmul(matmul(flat(root), flat(s2)), flat(root)).data
    middle = [[0.5 * (middle[i * size + j] + middle[j * size + i]) for j in range(size)] for i in range(size)]
    values, _ = symmetric_eigen(middle, tolerance=1e-12)
    trace_root = sum(math.sqrt(max(0.0, value)) for value in values)
    mean_term = sum((a - b) ** 2 for a, b in zip(mu1, mu2))
    return mean_term + sum(s1[i][i] + s2[i][i] for i in range(size)) - 2.0 * trace_root


class Judge:
    """A trained feature net plus the real test images' Gaussian, ready to score samples."""

    def __init__(self, dataset: str, *, seed: int = 0) -> None:
        self.model = train_feature_net(dataset, seed=seed)
        images, labels = load_split(dataset, Path(DATASETS[dataset][0]), "test", None)
        features, classes = run_in_batches(self.model, [SIGNED[b] for image in images for b in image])
        self.accuracy = sum(c == y for c, y in zip(classes, labels)) / len(labels)
        self.real = gaussian(features)
        self.train_images = load_split(dataset, Path(DATASETS[dataset][0]), "train", None)[0]

    def score(self, samples: list[float]) -> dict:
        """FID against the test set, and how many samples fell in each class."""

        features, classes = run_in_batches(self.model, samples)
        return {"fid": frechet_distance(gaussian(features), self.real),
                "class_counts": [classes.count(c) for c in range(10)]}

    def floor(self, count: int, seed: int = 0) -> float:
        """FID of ``count`` real training images: the best a perfect model could score with that many samples."""

        chosen = random.Random(seed).sample(range(len(self.train_images)), count)
        return self.score([SIGNED[b] for i in chosen for b in self.train_images[i]])["fid"]
