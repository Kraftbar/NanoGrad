"""Text classification on AG News with fastText's model, NanoGrad vs PyTorch.

AG News (Zhang, Zhao and LeCun, 2015) is 127,600 news headlines with a
short description, each labelled World, Sports, Business or Sci/Tech
(30,000 training and 1,900 test articles per class). fastText (Joulin et
al., 2016, "Bag of Tricks for Efficient Text Classification") showed
that a very simple model is a strong baseline on it:

    embed every word of the text, average the embeddings, apply a linear layer

10 numbers per word, nothing about word order except, optionally, word
pairs ("bigrams") hashed into extra embedding rows. The paper reports 91.5%
test accuracy with words and 92.5% with bigrams added.

Training here is minibatch SGD (batch 32, learning rate decaying linearly
to zero over 5 epochs, as fastText's schedule does), not fastText's one
example at a time; PyTorch runs the same model (``nn.EmbeddingBag``) from
the same weights on the same batches.

    python3 text_classify.py                       # words
    python3 text_classify.py --bigrams             # words and word pairs
    .venv/bin/python text_classify.py --frameworks nanograd torch

Data: data/ag_news/{train,test}.csv (see data/ag_news/README.md).
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import zlib
from collections import Counter
from pathlib import Path
from time import perf_counter

from losses import softmax_cross_entropy
from optim import TensorSGD
from tensor import Tensor, gather_rows, no_grad, scatter_add_rows
from tensor_nn import TensorLinear, TensorModule


CLASSES = ("World", "Sports", "Business", "Sci/Tech")
TOKEN = re.compile(r"[a-z0-9]+|[^a-z0-9\s]")


def tokenize(text: str) -> list[str]:
    """Lowercase words and punctuation marks (fastText's preprocessing for AG News)."""

    return TOKEN.findall(text.lower().replace("\\n", " ").replace("\\", " "))


def load(path: str | Path) -> list[tuple[list[str], int]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [(tokenize(title + " " + description), int(label) - 1) for label, title, description in csv.reader(handle)]


class Features:
    """Word ids from the training vocabulary, plus hashed word-pair ids."""

    def __init__(self, examples, *, bigrams: bool, buckets: int) -> None:
        counts = Counter(word for words, _ in examples for word in words)
        self.words = {word: index for index, (word, _) in enumerate(counts.most_common())}
        self.bigrams, self.buckets = bigrams, buckets
        self.size = len(self.words) + (buckets if bigrams else 0)

    def ids(self, words: list[str]) -> list[int]:
        known = [self.words[word] for word in words if word in self.words]
        if self.bigrams:
            known += [len(self.words) + zlib.crc32(f"{a} {b}".encode()) % self.buckets
                      for a, b in zip(words, words[1:])]
        return known or [0]


class BagOfEmbeddings(TensorModule):
    """fastText: the mean of a text's embeddings, then a linear layer.

    The mean over each text is two existing ops: gather the rows of every
    id in the batch, then scatter-add them onto their text's row and divide
    by the text's length (PyTorch's ``nn.EmbeddingBag(mode="mean")``).
    """

    def __init__(self, vocab_size: int, classes: int, *, dim: int = 10, seed: int = 0) -> None:
        rng = random.Random(seed)
        bound = 1.0 / dim                                            # fastText's initialization
        self.embedding = Tensor([rng.uniform(-bound, bound) for _ in range(vocab_size * dim)], (vocab_size, dim),
                                requires_grad=True)
        self.output = TensorLinear(dim, classes, weight=Tensor.zeros((classes, dim), requires_grad=True))

    def __call__(self, texts: list[list[int]]) -> Tensor:
        ids = [token for text in texts for token in text]
        owners = [row for row, text in enumerate(texts) for _ in text]
        sums = scatter_add_rows(gather_rows(self.embedding, ids), owners, len(texts))     # (B, dim)
        means = sums * Tensor([1.0 / len(text) for text in texts], (len(texts), 1))
        return self.output(means)

    def parameters(self) -> list[Tensor]:
        return [self.embedding, *self.output.parameters()]


class NanoGradRunner:
    def __init__(self, model: BagOfEmbeddings) -> None:
        self.model = model
        self.optimizer = TensorSGD(model.parameters(), lr=1.0)

    def train_step(self, texts, labels, lr: float) -> float:
        self.optimizer.zero_grad()
        loss = softmax_cross_entropy(self.model(texts), labels)
        loss.backward()
        self.optimizer.lr = lr
        self.optimizer.step()
        return loss.item()

    def predict(self, texts) -> list[int]:
        with no_grad():
            scores = self.model(texts)
        classes = scores.shape[1]
        return [max(range(classes), key=lambda c: scores.data[row * classes + c]) for row in range(len(texts))]

    def weights(self) -> list[list[float]]:
        return [list(parameter.data) for parameter in self.model.parameters()]


class TorchRunner:
    def __init__(self, model: BagOfEmbeddings) -> None:
        import torch

        self.torch = torch
        vocab, dim = model.embedding.shape
        self.bag = torch.nn.EmbeddingBag(vocab, dim, mode="mean", dtype=torch.float64)
        self.linear = torch.nn.Linear(dim, model.output.weight.shape[0], dtype=torch.float64)
        with torch.no_grad():
            self.bag.weight.copy_(torch.tensor(list(model.embedding.data), dtype=torch.float64).reshape(vocab, dim))
            self.linear.weight.zero_()
            self.linear.bias.zero_()
        self.parameters = [self.bag.weight, self.linear.weight, self.linear.bias]

    def _scores(self, texts):
        torch = self.torch
        flat = torch.tensor([token for text in texts for token in text])
        offsets = torch.tensor([0] + [len(text) for text in texts[:-1]]).cumsum(0)
        return self.linear(self.bag(flat, offsets))

    def train_step(self, texts, labels, lr: float) -> float:
        torch = self.torch
        for parameter in self.parameters:
            parameter.grad = None
        loss = torch.nn.functional.cross_entropy(self._scores(texts), torch.tensor(labels))
        loss.backward()
        with torch.no_grad():
            for parameter in self.parameters:
                parameter.add_(parameter.grad, alpha=-lr)
        return loss.item()

    def predict(self, texts) -> list[int]:
        with self.torch.no_grad():
            return self._scores(texts).argmax(dim=1).tolist()

    def weights(self) -> list[list[float]]:
        return [parameter.detach().reshape(-1).tolist() for parameter in self.parameters]


def train(runner, train_set, test_set, args) -> dict:
    rng = random.Random(args.seed)
    order = list(range(len(train_set)))
    steps = args.epochs * ((len(order) + args.batch_size - 1) // args.batch_size)
    step, start, recent = 0, perf_counter(), []
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(order)
        for first in range(0, len(order), args.batch_size):
            batch = [train_set[index] for index in order[first : first + args.batch_size]]
            lr = args.lr * (1.0 - step / steps)                       # linear decay to zero
            recent.append(runner.train_step([ids for ids, _ in batch], [label for _, label in batch], lr))
            step += 1
        print(f"  epoch {epoch}: train loss {sum(recent) / len(recent):.4f}, "
              f"{perf_counter() - start:.0f} s", flush=True)
        recent = []
    seconds = perf_counter() - start
    correct = 0
    for first in range(0, len(test_set), 1000):
        chunk = test_set[first : first + 1000]
        correct += sum(p == label for p, (_, label) in zip(runner.predict([ids for ids, _ in chunk]), chunk))
    return {"accuracy": correct / len(test_set), "ms_per_step": 1000 * seconds / step}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="data/ag_news")
    parser.add_argument("--frameworks", nargs="+", choices=("nanograd", "torch"), default=["nanograd"])
    parser.add_argument("--bigrams", action="store_true")
    parser.add_argument("--buckets", type=int, default=1 << 20, help="hash buckets for word pairs")
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2.0)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    train_texts = load(Path(args.data) / "train.csv")[: args.train_limit]
    test_texts = load(Path(args.data) / "test.csv")
    features = Features(train_texts, bigrams=args.bigrams, buckets=args.buckets)
    train_set = [(features.ids(words), label) for words, label in train_texts]
    test_set = [(features.ids(words), label) for words, label in test_texts]
    print(f"{len(train_set)} training and {len(test_set)} test texts; {len(features.words)} words"
          + (f" + {args.buckets} word-pair buckets" if args.bigrams else ""), flush=True)

    results, runners = {}, {}
    for framework in args.frameworks:
        model = BagOfEmbeddings(features.size, len(CLASSES), dim=args.dim, seed=args.seed)
        if framework == "torch":
            import torch

            torch.set_num_threads(1)
            runners[framework] = TorchRunner(model)
        else:
            runners[framework] = NanoGradRunner(model)
        print(framework, flush=True)
        results[framework] = train(runners[framework], train_set, test_set, args)
        print(f"{framework:8s}: test accuracy {100 * results[framework]['accuracy']:.2f}%, "
              f"{results[framework]['ms_per_step']:.1f} ms/step", flush=True)
    if len(runners) == 2:
        difference = max(abs(a - b) for x, y in zip(runners["nanograd"].weights(), runners["torch"].weights())
                         for a, b in zip(x, y))
        print(f"largest weight difference after training: {difference:.1e}")
        results["max_weight_difference"] = difference
    return results


if __name__ == "__main__":
    main()
