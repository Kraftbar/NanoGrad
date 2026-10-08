"""Recommendation: neural collaborative filtering on MovieLens-1M, NanoGrad vs PyTorch.

A recommender sees only who interacted with what: 6,040 users who rated
3,706 movies (the stars are ignored, a rating just means "watched"). It
must rank the movies a user has not seen yet. Neural collaborative
filtering (He et al., 2017) gives every user and every movie a learned
vector and scores a pair from them:

    GMF    score = h . (p_u * q_i)                      matrix factorization, elementwise product
    MLP    score = h . MLP([p_u, q_i])                  the two vectors side by side, through layers
    NeuMF  score = h . [p_u * q_i, MLP([p'_u, q'_i])]   both, with separate vectors for each half

Training treats it as classification: every (user, movie) seen is a 1,
and 4 movies the user never saw are sampled for each as 0s, fresh every
epoch; binary cross entropy on the scores.

Evaluation is the paper's leave-one-out protocol: each user's latest
movie was held out; rank it among 99 movies the user never saw. HR@10 is
how often it lands in the top 10, NDCG@10 also rewards placing it higher
(1 / log2(rank + 2)). The paper reports HR@10 0.70 / NDCG@10 0.42 for
GMF with 8 factors, 0.69 / 0.42 for MLP, 0.73 / 0.45 for NeuMF with
pre-training (about 0.70 without).

    python3 recommend.py --model gmf
    python3 recommend.py --model neumf --epochs 20
    .venv/bin/python recommend.py --model mlp --frameworks nanograd torch --epochs 2

Data: data/ml-1m/ (see data/ml-1m/README.md).
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from time import perf_counter

from losses import binary_cross_entropy_with_logits
from optim import TensorAdam
from tensor import Tensor, concat, gather_rows, no_grad
from tensor_nn import TensorLinear, TensorModule


class Ratings:
    """The NCF split: training pairs, one held-out item per user, its 99 negatives."""

    def __init__(self, folder: str | Path, *, limit_users: int | None = None) -> None:
        folder = Path(folder)
        self.train: list[tuple[int, int]] = []
        with (folder / "ml-1m.train.rating").open() as handle:
            for line in handle:
                user, item = line.split("\t")[:2]
                self.train.append((int(user), int(item)))
        self.test: list[tuple[int, list[int]]] = []          # (user, [held-out item, 99 negatives])
        with (folder / "ml-1m.test.negative").open() as handle:
            for line in handle:
                fields = line.split()
                user, item = fields[0].strip("()").split(",")
                self.test.append((int(user), [int(item)] + [int(f) for f in fields[1:]]))
        if limit_users is not None:
            self.train = [(u, i) for u, i in self.train if u < limit_users]
            self.test = [entry for entry in self.test if entry[0] < limit_users]
        self.users = 1 + max(max(u for u, _ in self.train), max(u for u, _ in self.test))
        self.items = 1 + max(max(i for _, i in self.train), max(max(items) for _, items in self.test))
        self.seen: dict[int, set[int]] = {}
        for user, item in self.train:
            self.seen.setdefault(user, set()).add(item)

    def epoch(self, negatives: int, rng: random.Random) -> list[tuple[int, int, float]]:
        """Every training pair as a 1, plus ``negatives`` unseen items per pair as 0s, shuffled."""

        examples = []
        for user, item in self.train:
            examples.append((user, item, 1.0))
            seen = self.seen[user]
            for _ in range(negatives):
                other = rng.randrange(self.items)
                while other in seen:
                    other = rng.randrange(self.items)
                examples.append((user, other, 0.0))
        rng.shuffle(examples)
        return examples


def embedding(rows: int, dim: int, rng: random.Random) -> Tensor:
    return Tensor([rng.gauss(0.0, 0.01) for _ in range(rows * dim)], (rows, dim), requires_grad=True)


class NCF(TensorModule):
    """GMF, MLP or NeuMF (both, side by side), as in the paper's code."""

    def __init__(self, users: int, items: int, *, model: str = "neumf", factors: int = 8,
                 layers: tuple[int, ...] = (64, 32, 16, 8), seed: int = 0) -> None:
        rng = random.Random(seed)
        self.model = model
        self.gmf = model in ("gmf", "neumf")
        self.mlp = model in ("mlp", "neumf")
        width = 0
        if self.gmf:
            self.gmf_user, self.gmf_item = embedding(users, factors, rng), embedding(items, factors, rng)
            width += factors
        if self.mlp:
            # The first layer's input is the two vectors side by side, so each is half of it.
            self.mlp_user, self.mlp_item = embedding(users, layers[0] // 2, rng), embedding(items, layers[0] // 2, rng)
            self.hidden = [TensorLinear(a, b, seed=rng.randrange(1 << 30)) for a, b in zip(layers, layers[1:])]
            width += layers[-1]
        self.output = TensorLinear(width, 1, seed=rng.randrange(1 << 30))

    def __call__(self, users: list[int], items: list[int]) -> Tensor:
        """A score (logit) per (user, item) pair: shape (B, 1)."""

        parts = []
        if self.gmf:
            parts.append(gather_rows(self.gmf_user, users) * gather_rows(self.gmf_item, items))
        if self.mlp:
            x = concat([gather_rows(self.mlp_user, users), gather_rows(self.mlp_item, items)], axis=1)
            for layer in self.hidden:
                x = layer(x).relu()
            parts.append(x)
        return self.output(parts[0] if len(parts) == 1 else concat(parts, axis=1))

    def parameters(self) -> list[Tensor]:
        found = []
        if self.gmf:
            found += [self.gmf_user, self.gmf_item]
        if self.mlp:
            found += [self.mlp_user, self.mlp_item] + [p for layer in self.hidden for p in layer.parameters()]
        return found + self.output.parameters()


class NanoGradRunner:
    def __init__(self, model: NCF, lr: float) -> None:
        self.model = model
        self.optimizer = TensorAdam(model.parameters(), lr=lr)

    def train_step(self, users, items, labels) -> float:
        self.optimizer.zero_grad()
        loss = binary_cross_entropy_with_logits(self.model(users, items), Tensor(labels, (len(labels), 1)))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def scores(self, users, items) -> list[float]:
        with no_grad():
            return list(self.model(users, items).data)

    def weights(self) -> list[list[float]]:
        return [list(parameter.data) for parameter in self.model.parameters()]


class TorchRunner:
    """The same network in PyTorch, started from NanoGrad's weights."""

    def __init__(self, model: NCF, lr: float) -> None:
        import torch

        self.torch = torch
        self.model = model
        self.parameters = [torch.tensor(list(p.data), dtype=torch.float64).reshape(p.shape).requires_grad_()
                           for p in model.parameters()]
        self.optimizer = torch.optim.Adam(self.parameters, lr=lr)

    def _scores(self, users, items):
        torch, model = self.torch, self.model
        params = iter(self.parameters)
        users, items = torch.tensor(users), torch.tensor(items)
        parts = []
        if model.gmf:
            p, q = next(params), next(params)
            parts.append(p[users] * q[items])
        if model.mlp:
            p, q = next(params), next(params)
            x = torch.cat([p[users], q[items]], dim=1)
            for _ in model.hidden:
                weight, bias = next(params), next(params)
                x = torch.relu(x @ weight.T + bias)
            parts.append(x)
        weight, bias = next(params), next(params)
        return torch.cat(parts, dim=1) @ weight.T + bias

    def train_step(self, users, items, labels) -> float:
        torch = self.torch
        self.optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            self._scores(users, items), torch.tensor(labels, dtype=torch.float64).reshape(-1, 1))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def scores(self, users, items) -> list[float]:
        with self.torch.no_grad():
            return self._scores(users, items).reshape(-1).tolist()

    def weights(self) -> list[list[float]]:
        return [parameter.detach().reshape(-1).tolist() for parameter in self.parameters]


def evaluate(runner, ratings: Ratings, k: int = 10) -> tuple[float, float]:
    """HR@k and NDCG@k: the held-out item's rank among itself and its 99 negatives."""

    hits = ndcg = 0.0
    for first in range(0, len(ratings.test), 500):
        chunk = ratings.test[first : first + 500]
        users = [user for user, items in chunk for _ in items]
        items = [item for _, items in chunk for item in items]
        scores = runner.scores(users, items)
        offset = 0
        for _, candidates in chunk:
            row = scores[offset : offset + len(candidates)]
            offset += len(candidates)
            rank = sum(score > row[0] for score in row[1:])        # items scored above the held-out one
            if rank < k:
                hits += 1.0
                ndcg += 1.0 / math.log2(rank + 2)
    return hits / len(ratings.test), ndcg / len(ratings.test)


def train(runner, ratings: Ratings, args) -> dict:
    rng = random.Random(args.seed)
    history, steps, training, start = [], 0, 0.0, perf_counter()
    hr, ndcg = evaluate(runner, ratings)
    print(f"  untrained: HR@10 {hr:.4f}, NDCG@10 {ndcg:.4f}", flush=True)
    for epoch in range(1, args.epochs + 1):
        examples = ratings.epoch(args.negatives, rng)
        if args.step_limit:
            examples = examples[: args.step_limit * args.batch_size]
        total, began = 0.0, perf_counter()
        for first in range(0, len(examples), args.batch_size):
            batch = examples[first : first + args.batch_size]
            total += runner.train_step([u for u, _, _ in batch], [i for _, i, _ in batch], [y for _, _, y in batch])
            steps += 1
        training += perf_counter() - began
        batches = (len(examples) + args.batch_size - 1) // args.batch_size
        hr, ndcg = evaluate(runner, ratings)
        history.append((total / batches, hr, ndcg))
        print(f"  epoch {epoch:2d}: loss {total / batches:.4f}, HR@10 {hr:.4f}, NDCG@10 {ndcg:.4f}, "
              f"{perf_counter() - start:.0f} s", flush=True)
    best = max(history, key=lambda h: h[1])
    return {"history": history, "hr": best[1], "ndcg": best[2], "seconds": perf_counter() - start,
            "ms_per_step": 1000 * training / max(1, steps)}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="data/ml-1m")
    parser.add_argument("--model", choices=("gmf", "mlp", "neumf"), default="neumf")
    parser.add_argument("--frameworks", nargs="+", choices=("nanograd", "torch"), default=["nanograd"])
    parser.add_argument("--factors", type=int, default=8)
    parser.add_argument("--layers", type=int, nargs="+", default=[64, 32, 16, 8])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--step-limit", type=int, default=0, help="stop each epoch after this many steps")
    parser.add_argument("--limit-users", type=int, help="use only the first N users (quick runs)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    ratings = Ratings(args.data, limit_users=args.limit_users)
    print(f"{len(ratings.train)} training pairs, {ratings.users} users, {ratings.items} items", flush=True)
    results, runners = {}, {}
    for framework in args.frameworks:
        model = NCF(ratings.users, ratings.items, model=args.model, factors=args.factors,
                    layers=tuple(args.layers), seed=args.seed)
        if framework == "torch":
            import torch

            torch.set_num_threads(1)
            runners[framework] = TorchRunner(model, args.lr)
        else:
            runners[framework] = NanoGradRunner(model, args.lr)
        print(f"{framework} {args.model}", flush=True)
        results[framework] = result = train(runners[framework], ratings, args)
        print(f"{framework:8s} {args.model}: best HR@10 {result['hr']:.4f}, NDCG@10 {result['ndcg']:.4f}, "
              f"{result['ms_per_step']:.2f} ms/step, {result['seconds'] / 60:.0f} min", flush=True)
    if len(runners) == 2:
        difference = max(abs(a - b) for x, y in zip(runners["nanograd"].weights(), runners["torch"].weights())
                         for a, b in zip(x, y))
        print(f"largest weight difference after training: {difference:.1e}")
        results["max_weight_difference"] = difference
    return results


if __name__ == "__main__":
    main()
