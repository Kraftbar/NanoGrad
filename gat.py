"""Graph attention network (GAT) on the Cora citation graph.

A GCN (gcn.py) averages a node's neighbours with fixed weights. A graph
attention layer (Velickovic et al., 2018) learns how much each neighbour
matters, separately for every edge:

    z_i      = W h_i                                            every node, projected
    e_ij     = LeakyReLU(a_src . z_j + a_dst . z_i)             a score per edge j -> i
    alpha_ij = exp(e_ij) / sum over neighbours k of i of exp(e_ik)
    h_i'     = sum over neighbours j of alpha_ij z_j

with several heads side by side, like transformer attention restricted to
the graph's edges. The softmax runs over each node's own neighbours, a
different number for every node; it is built from ops the core already
has: gather each edge's endpoint values, exp, scatter-add the sums back
onto the nodes, gather them again to divide.

The paper's recipe: 8 heads of 8 features with ELU, then one attention
head to the 7 classes; dropout 0.6 on the inputs and the attention
weights; Adam at lr 0.005 with L2 5e-4. It reports 83.0% on Cora.

    python3 gat.py --seeds 3
"""

from __future__ import annotations

import argparse
import math
import random
from time import perf_counter

from gcn import Graph, accuracy
from losses import softmax_cross_entropy
from optim import TensorAdam
from tensor import Tensor, concat, gather_rows, no_grad, scatter_add_rows, sparse_matmul
from tensor_nn import LeakyReLU, TensorDropout, TensorModule


def elu(x: Tensor) -> Tensor:
    """x for x > 0, e^x - 1 below: relu(x) + e^(min(x, 0)) - 1."""

    return x.relu() + (-((-x).relu())).exp() - 1.0


class GraphAttention(TensorModule):
    """One multi-head graph attention layer over a fixed edge list."""

    def __init__(self, inputs: int, outputs: int, *, heads: int, dropout: float, rng: random.Random) -> None:
        def glorot(rows: int, cols: int) -> Tensor:
            bound = math.sqrt(6.0 / (rows + cols))
            return Tensor([rng.uniform(-bound, bound) for _ in range(rows * cols)], (rows, cols), requires_grad=True)

        self.heads, self.outputs = heads, outputs
        self.weight = glorot(inputs, heads * outputs)                 # all heads' W side by side
        self.attend_source = glorot(heads * outputs, 1)              # a_src for every head, stacked
        self.attend_target = glorot(heads * outputs, 1)
        self.bias = Tensor.zeros((heads * outputs,), requires_grad=True)
        self.leaky = LeakyReLU(0.2)
        self.dropout = TensorDropout(dropout, seed=rng.randrange(1 << 30))

    def __call__(self, h, sources: list[int], targets: list[int], nodes: int) -> Tensor:
        """New features (nodes, heads * outputs); edge k goes sources[k] -> targets[k]."""

        heads, outputs = self.heads, self.outputs
        z = sparse_matmul(h, self.weight) if not isinstance(h, Tensor) else h @ self.weight   # (N, H*F)
        # Per node and head: a_src . z and a_dst . z, as one product each with
        # block-diagonal attention vectors (head k only sees its own F features).
        score_source = (z * self.attend_source.reshape((1, heads * outputs))).reshape((nodes, heads, outputs)).sum(axis=-1)
        score_target = (z * self.attend_target.reshape((1, heads * outputs))).reshape((nodes, heads, outputs)).sum(axis=-1)
        edges = len(sources)
        scores = self.leaky(gather_rows(score_source, sources) + gather_rows(score_target, targets))   # (E, H)
        # Softmax over each target node's incoming edges. Any constant can be
        # subtracted first without changing it; the largest score keeps exp finite.
        weights = (scores - max(scores.data)).exp()
        totals = scatter_add_rows(weights, targets, nodes)                                   # (N, H)
        alpha = self.dropout(weights / gather_rows(totals, targets))                         # (E, H)
        # Each edge carries alpha * z_source to its target, head by head.
        messages = gather_rows(z, sources).reshape((edges, heads, outputs)) * alpha.reshape((edges, heads, 1))
        return scatter_add_rows(messages.reshape((edges, heads * outputs)), targets, nodes) + self.bias

    def parameters(self) -> list[Tensor]:
        return [self.weight, self.attend_source, self.attend_target, self.bias]


class GAT(TensorModule):
    def __init__(self, features: int, classes: int, *, hidden: int = 8, heads: int = 8, dropout: float = 0.6,
                 seed: int = 0) -> None:
        rng = random.Random(seed)
        self.layer1 = GraphAttention(features, hidden, heads=heads, dropout=dropout, rng=rng)
        self.layer2 = GraphAttention(heads * hidden, classes, heads=1, dropout=dropout, rng=rng)
        self.input_dropout = TensorDropout(dropout, seed=seed + 1)
        self.hidden_dropout = TensorDropout(dropout, seed=seed + 2)

    def __call__(self, graph: Graph, features: Tensor, edges: tuple[list[int], list[int]]) -> Tensor:
        nodes = len(graph.labels)
        hidden = elu(self.layer1(self.input_dropout(features), *edges, nodes))
        return self.layer2(self.hidden_dropout(hidden), *edges, nodes)

    def parameters(self) -> list[Tensor]:
        return [*self.layer1.parameters(), *self.layer2.parameters()]


def edge_list(graph: Graph) -> tuple[list[int], list[int]]:
    """Every link in both directions plus self-loops, as (sources, targets)."""

    matrix = graph.adjacency
    sources, targets = [], []
    for node in range(matrix.shape[0]):
        for e in range(matrix.row_start[node], matrix.row_start[node + 1]):
            sources.append(matrix.cols[e])
            targets.append(node)
    return sources, targets


def train(graph: Graph, args, seed: int) -> dict:
    model = GAT(graph.features.shape[1], len(graph.classes), hidden=args.hidden, heads=args.heads,
                dropout=args.dropout, seed=seed)
    features = Tensor(graph.features.to_dense(), graph.features.shape)
    edges = edge_list(graph)
    optimizer = TensorAdam(model.parameters(), lr=args.lr, l2=args.weight_decay)
    train_nodes, labels = list(graph.train), [graph.labels[n] for n in graph.train]
    classes, best, best_test, patience, start = len(graph.classes), -1.0, 0.0, 0, perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = softmax_cross_entropy(gather_rows(model(graph, features, edges), train_nodes), labels)
        loss.backward()
        optimizer.step()
        model.eval()
        with no_grad():
            scores = list(model(graph, features, edges).data)
        validation = accuracy(scores, graph.labels, graph.validation, classes)
        if validation > best:                     # early stopping on validation accuracy
            best, best_test, patience = validation, accuracy(scores, graph.labels, graph.test, classes), 0
        else:
            patience += 1
            if patience >= args.patience:
                break
    return {"validation": best, "test": best_test, "epochs": epoch, "seconds": perf_counter() - start}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="data/cora")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    args = parser.parse_args(argv)
    graph = Graph(args.data)
    tests = []
    for seed in range(args.seeds):
        result = train(graph, args, seed)
        tests.append(result["test"])
        print(f"seed {seed}: validation {100 * result['validation']:.1f}%, test {100 * result['test']:.1f}% "
              f"(stopped after {result['epochs']} epochs, {result['seconds']:.0f} s)", flush=True)
    mean = sum(tests) / len(tests)
    spread = math.sqrt(sum((t - mean) ** 2 for t in tests) / max(1, len(tests) - 1))
    print(f"GAT: test accuracy {100 * mean:.1f}% +- {100 * spread:.1f} over {len(tests)} seeds")
    return {"mean": mean, "std": spread, "tests": tests}


if __name__ == "__main__":
    main()
