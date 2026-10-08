"""Graph convolutional network (GCN) on the Cora citation graph.

Cora is 2,708 machine-learning papers, each a node with 1,433 binary
features (which words of a dictionary the paper uses) and one of 7 topics;
5,429 citations connect them. Given the topics of just 140 papers, predict
the rest. Kipf and Welling's GCN (2017) made this the standard benchmark
for graph neural networks.

A graph convolution mixes every node's features with its neighbours':

    H' = A_hat @ (H W) + b         A_hat = D^-1 (A + I)

A is the adjacency matrix (A + I adds each node to its own neighbourhood)
and D^-1 divides each row by its node's degree, so A_hat averages over the
neighbourhood. Two such layers (ReLU and dropout between them) let each
prediction depend on everything within two citations. A_hat and the
features are mostly zeros, so both are ``SparseMatrix``es: a product with
one costs one row operation per nonzero.

The recipe is PyTorch's reference implementation (tkipf/pygcn): row
normalization, 16 hidden units, dropout 0.5, Adam (lr 0.01, L2 5e-4), 200
full-graph steps, papers 0-139 to train on, 200-499 to validate, 500-1499
to test.

    python3 gcn.py                                   # 10 seeds, NanoGrad
    .venv/bin/python gcn.py --frameworks nanograd torch
    .venv/bin/python gcn.py --compare                # same weights, no dropout, both

Data: data/cora/cora.content and cora.cites (see data/cora/README.md).
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from time import perf_counter

from losses import softmax_cross_entropy
from optim import TensorAdam
from tensor import SparseMatrix, Tensor, gather_rows, no_grad, sparse_matmul
from tensor_nn import TensorDropout, TensorModule


class Graph:
    """Cora's features, labels and normalized adjacency, in file order."""

    def __init__(self, folder: str | Path) -> None:
        folder = Path(folder)
        rows = [line.split() for line in (folder / "cora.content").read_text().splitlines() if line.strip()]
        ids = {row[0]: index for index, row in enumerate(rows)}
        self.classes = sorted({row[-1] for row in rows})
        self.labels = [self.classes.index(row[-1]) for row in rows]
        count, width = len(rows), len(rows[0]) - 2

        # Features: each paper's word indicators, divided by how many it has.
        entries = []
        for node, row in enumerate(rows):
            words = [column for column, value in enumerate(row[1:-1]) if value != "0"]
            entries += [(node, column, 1.0 / len(words)) for column in words]
        self.features = SparseMatrix((count, width), entries)

        # Adjacency: citations in both directions, plus self-loops, rows averaged.
        neighbours = [{node} for node in range(count)]
        for line in (folder / "cora.cites").read_text().splitlines():
            if line.strip():
                cited, citing = (ids[name] for name in line.split())
                neighbours[cited].add(citing)
                neighbours[citing].add(cited)
        self.adjacency = SparseMatrix((count, count), [
            (node, other, 1.0 / len(around)) for node, around in enumerate(neighbours) for other in around
        ])
        self.edges = sum(len(around) - 1 for around in neighbours) // 2
        self.train, self.validation, self.test = range(140), range(200, 500), range(500, 1500)


class GraphConvolution(TensorModule):
    """H' = A_hat @ (H W) + b, initialized like pygcn: uniform(+-1/sqrt(outputs))."""

    def __init__(self, inputs: int, outputs: int, *, rng: random.Random) -> None:
        bound = 1 / math.sqrt(outputs)
        self.weight = Tensor([rng.uniform(-bound, bound) for _ in range(inputs * outputs)], (inputs, outputs),
                             requires_grad=True)
        self.bias = Tensor([rng.uniform(-bound, bound) for _ in range(outputs)], (outputs,), requires_grad=True)

    def __call__(self, adjacency: SparseMatrix, hidden: Tensor | SparseMatrix) -> Tensor:
        support = sparse_matmul(hidden, self.weight) if isinstance(hidden, SparseMatrix) else hidden @ self.weight
        return sparse_matmul(adjacency, support) + self.bias

    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]


class GCN(TensorModule):
    def __init__(self, features: int, classes: int, *, hidden: int = 16, dropout: float = 0.5, seed: int = 0) -> None:
        rng = random.Random(seed)
        self.layer1 = GraphConvolution(features, hidden, rng=rng)
        self.layer2 = GraphConvolution(hidden, classes, rng=rng)
        self.dropout = TensorDropout(dropout, seed=seed + 1)

    def __call__(self, graph: Graph) -> Tensor:
        """Class scores for every node: (nodes, classes)."""

        hidden = self.layer1(graph.adjacency, graph.features).relu()
        return self.layer2(graph.adjacency, self.dropout(hidden))

    def parameters(self) -> list[Tensor]:
        return [*self.layer1.parameters(), *self.layer2.parameters()]


def accuracy(scores: list[float], labels: list[int], nodes, classes: int) -> float:
    correct = 0
    for node in nodes:
        row = scores[node * classes : (node + 1) * classes]
        correct += max(range(classes), key=row.__getitem__) == labels[node]
    return correct / len(nodes)


class NanoGradRunner:
    def __init__(self, model: GCN, graph: Graph, args) -> None:
        self.model, self.graph = model, graph
        self.optimizer = TensorAdam(model.parameters(), lr=args.lr, l2=args.weight_decay)
        self.train_labels = [graph.labels[node] for node in graph.train]

    def train_step(self) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        scores = self.model(self.graph)
        loss = softmax_cross_entropy(gather_rows(scores, list(self.graph.train)), self.train_labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def scores(self) -> list[float]:
        self.model.eval()
        with no_grad():
            return list(self.model(self.graph).data)

    def weights(self) -> list[list[float]]:
        return [list(parameter.data) for parameter in self.model.parameters()]


class TorchRunner:
    """pygcn's model in PyTorch (torch.sparse.mm), from the NanoGrad model's weights."""

    def __init__(self, model: GCN, graph: Graph, args, *, dropout: float) -> None:
        import torch

        self.torch, self.graph, self.dropout = torch, graph, dropout

        def sparse(matrix: SparseMatrix):
            rows = [r for r in range(matrix.shape[0]) for _ in range(matrix.row_start[r], matrix.row_start[r + 1])]
            index = torch.tensor([rows, list(matrix.cols)])
            return torch.sparse_coo_tensor(index, torch.tensor(list(matrix.values), dtype=torch.float64),
                                           matrix.shape, check_invariants=False).coalesce()

        self.adjacency, self.features = sparse(graph.adjacency), sparse(graph.features)
        self.parameters = [torch.tensor(list(p.data), dtype=torch.float64).reshape(p.shape).requires_grad_()
                           for p in model.parameters()]
        self.optimizer = torch.optim.Adam(self.parameters, lr=args.lr, weight_decay=args.weight_decay)
        self.train_nodes = torch.tensor(list(graph.train))
        self.train_labels = torch.tensor([graph.labels[node] for node in graph.train])
        self.training = True

    def _scores(self):
        torch = self.torch
        w1, b1, w2, b2 = self.parameters
        hidden = torch.relu(torch.sparse.mm(self.adjacency, torch.sparse.mm(self.features, w1)) + b1)
        hidden = torch.nn.functional.dropout(hidden, self.dropout, training=self.training)
        return torch.sparse.mm(self.adjacency, hidden @ w2) + b2

    def train_step(self) -> float:
        self.training = True
        self.optimizer.zero_grad()
        loss = self.torch.nn.functional.cross_entropy(self._scores()[self.train_nodes], self.train_labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def scores(self) -> list[float]:
        self.training = False
        with self.torch.no_grad():
            return self._scores().reshape(-1).tolist()

    def weights(self) -> list[list[float]]:
        return [p.detach().reshape(-1).tolist() for p in self.parameters]


def train(runner, graph: Graph, epochs: int) -> dict:
    classes = len(graph.classes)
    start = perf_counter()
    for _ in range(epochs):
        loss = runner.train_step()
    seconds = perf_counter() - start
    scores = runner.scores()
    return {
        "loss": loss, "ms_per_step": 1000 * seconds / epochs,
        "validation": accuracy(scores, graph.labels, graph.validation, classes),
        "test": accuracy(scores, graph.labels, graph.test, classes),
    }


def make_runner(framework: str, graph: Graph, args, seed: int, *, dropout: float):
    model = GCN(graph.features.shape[1], len(graph.classes), hidden=args.hidden, dropout=dropout, seed=seed)
    if framework == "torch":
        import torch

        torch.manual_seed(seed)
        return TorchRunner(model, graph, args, dropout=dropout)
    return NanoGradRunner(model, graph, args)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="data/cora")
    parser.add_argument("--frameworks", nargs="+", choices=("nanograd", "torch"), default=["nanograd"])
    parser.add_argument("--compare", action="store_true", help="both frameworks, same weights, no dropout")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    args = parser.parse_args(argv)

    graph = Graph(args.data)
    print(f"{len(graph.labels)} papers, {graph.edges} citation links, {len(graph.classes)} topics, "
          f"{graph.features.shape[1]} features; train {len(graph.train)}, validate {len(graph.validation)}, "
          f"test {len(graph.test)}")
    if args.compare:
        import torch

        torch.set_num_threads(1)
        runners = {name: make_runner(name, graph, args, 0, dropout=0.0) for name in ("nanograd", "torch")}
        results = {name: train(runner, graph, args.epochs) for name, runner in runners.items()}
        for name, result in results.items():
            print(f"{name:8s}: test {100 * result['test']:.1f}%, loss {result['loss']:.6f}, "
                  f"{result['ms_per_step']:.2f} ms/step")
        difference = max(abs(a - b) for x, y in zip(runners["nanograd"].weights(), runners["torch"].weights())
                         for a, b in zip(x, y))
        print(f"largest weight difference after {args.epochs} steps: {difference:.1e}")
        return results

    summary = {}
    for framework in args.frameworks:
        if framework == "torch":
            import torch

            torch.set_num_threads(1)
        tests = []
        for seed in range(args.seeds):
            result = train(make_runner(framework, graph, args, seed, dropout=args.dropout), graph, args.epochs)
            tests.append(result["test"])
            print(f"{framework:8s} seed {seed}: validation {100 * result['validation']:.1f}%, "
                  f"test {100 * result['test']:.1f}%, {result['ms_per_step']:.2f} ms/step", flush=True)
        mean = sum(tests) / len(tests)
        spread = math.sqrt(sum((t - mean) ** 2 for t in tests) / max(1, len(tests) - 1))
        print(f"{framework:8s}: test accuracy {100 * mean:.1f}% +- {100 * spread:.1f} over {len(tests)} seeds")
        summary[framework] = {"mean": mean, "std": spread, "tests": tests}
    return summary


if __name__ == "__main__":
    main()
