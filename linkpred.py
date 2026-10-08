"""Link prediction: graph autoencoders (GAE and VGAE) on Cora, NanoGrad vs PyTorch.

The GCN (gcn.py) labels nodes. Link prediction asks which edges are
missing: hide 15% of Cora's 5,278 citations, learn from the rest, then
tell the hidden ones apart from as many random unconnected pairs. Kipf and
Welling's graph autoencoder (2016) embeds every node with a two-layer GCN
and scores a pair by the dot product of their embeddings:

    Z = GCN(X, A)                       (nodes, 16)
    p(edge i-j) = sigmoid(z_i . z_j)    all pairs at once: sigmoid(Z Z^T)

It is trained to reconstruct the whole adjacency matrix, 2,708^2 = 7.3
million pairs per step, by binary cross entropy. Edges are rare, so each
one is weighted by (non-edges / edges). The variational version (VGAE)
makes the GCN output a mean and a log standard deviation per node, samples
Z = mu + sigma * noise, and adds the KL divergence to N(0, I), as in a VAE.

The recipe is the authors' (tkipf/gae): a 32-unit then 16-unit GCN with
the symmetric normalization D^-1/2 (A + I) D^-1/2, raw binary features,
Adam at lr 0.01, 200 full-graph steps, 10% of edges to test and 5% to
validate. They report test AUC / average precision of 91.0 / 92.0 for
GAE and 91.4 / 92.6 for VGAE.

    python3 linkpred.py --model vgae --seeds 5
    .venv/bin/python linkpred.py --frameworks nanograd torch --epochs 20

Data: data/cora/ (see data/cora/README.md).
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from time import perf_counter

import kernels
from losses import binary_cross_entropy_with_logits
from optim import TensorAdam
from tensor import SparseMatrix, Tensor, gather_rows, no_grad, sparse_matmul
from tensor_nn import TensorModule


class CoraLinks:
    """Cora's features and undirected citation edges, split into train / validation / test."""

    def __init__(self, folder: str | Path, seed: int) -> None:
        folder = Path(folder)
        rows = [line.split() for line in (folder / "cora.content").read_text().splitlines() if line.strip()]
        ids = {row[0]: index for index, row in enumerate(rows)}
        self.nodes, width = len(rows), len(rows[0]) - 2
        self.features = SparseMatrix((self.nodes, width), [
            (node, column, 1.0) for node, row in enumerate(rows) for column, value in enumerate(row[1:-1])
            if value != "0"
        ])
        edges = set()
        for line in (folder / "cora.cites").read_text().splitlines():
            if line.strip():
                a, b = (ids[name] for name in line.split())
                if a != b:
                    edges.add((min(a, b), max(a, b)))
        edges = sorted(edges)
        rng = random.Random(seed)
        rng.shuffle(edges)
        tests, validations = len(edges) // 10, len(edges) // 20
        self.test_edges, self.validation_edges = edges[:tests], edges[tests : tests + validations]
        self.train_edges = edges[tests + validations :]
        known = set(edges)

        def non_edges(count: int) -> list[tuple[int, int]]:
            found = set()
            while len(found) < count:
                a, b = rng.randrange(self.nodes), rng.randrange(self.nodes)
                pair = (min(a, b), max(a, b))
                if a != b and pair not in known and pair not in found:
                    found.add(pair)
            return sorted(found)

        self.test_non_edges, self.validation_non_edges = non_edges(tests), non_edges(validations)
        self.edges = len(edges)

        # The encoder sees training edges only: D^-1/2 (A + I) D^-1/2.
        neighbours = [{node} for node in range(self.nodes)]
        for a, b in self.train_edges:
            neighbours[a].add(b)
            neighbours[b].add(a)
        degree = [len(around) for around in neighbours]
        self.adjacency = SparseMatrix((self.nodes, self.nodes), [
            (a, b, 1.0 / math.sqrt(degree[a] * degree[b])) for a, around in enumerate(neighbours) for b in around
        ])
        # Reconstruction targets: the same A + I, as positions in the flattened N x N matrix.
        self.positives = sorted(a * self.nodes + b for a, around in enumerate(neighbours) for b in around)
        targets = kernels.zeros(self.nodes * self.nodes)
        for position in self.positives:
            targets[position] = 1.0
        self.targets = Tensor(targets, (self.nodes * self.nodes, 1))


def glorot(rows: int, cols: int, rng: random.Random) -> Tensor:
    bound = math.sqrt(6.0 / (rows + cols))
    return Tensor([rng.uniform(-bound, bound) for _ in range(rows * cols)], (rows, cols), requires_grad=True)


class GraphAutoencoder(TensorModule):
    """GCN encoder (no biases, as in the paper's code); VGAE adds a log-sigma head."""

    def __init__(self, features: int, *, hidden: int = 32, latent: int = 16, variational: bool = True,
                 seed: int = 0) -> None:
        rng = random.Random(seed)
        self.variational = variational
        self.weight1 = glorot(features, hidden, rng)
        self.weight_mu = glorot(hidden, latent, rng)
        self.weight_sigma = glorot(hidden, latent, rng) if variational else None

    def encode(self, graph: CoraLinks) -> tuple[Tensor, Tensor | None]:
        hidden = sparse_matmul(graph.adjacency, sparse_matmul(graph.features, self.weight1)).relu()
        mu = sparse_matmul(graph.adjacency, hidden @ self.weight_mu)
        log_sigma = sparse_matmul(graph.adjacency, hidden @ self.weight_sigma) if self.variational else None
        return mu, log_sigma

    def loss(self, graph: CoraLinks, noise: Tensor | None) -> Tensor:
        """Weighted reconstruction cross entropy of all N^2 pairs (+ KL for VGAE)."""

        n = graph.nodes
        mu, log_sigma = self.encode(graph)
        z = mu + noise * log_sigma.exp() if self.variational else mu
        logits = (z @ z.T).reshape((n * n, 1))
        positives = len(graph.positives)
        # Every edge counts pos_weight times: once in the full sum, pos_weight - 1 more times here.
        pos_weight = (n * n - positives) / positives
        norm = n * n / (2.0 * (n * n - positives))
        full = binary_cross_entropy_with_logits(logits, graph.targets, reduction="sum")
        edges = binary_cross_entropy_with_logits(gather_rows(logits, graph.positives),
                                                 Tensor([1.0] * positives, (positives, 1)), reduction="sum")
        loss = (full + edges * (pos_weight - 1.0)) * (norm / (n * n))
        if self.variational:
            # KL(N(mu, sigma^2) || N(0, 1)), summed over the latent axis, averaged over nodes, over N again.
            kl = (log_sigma * 2.0 + 1.0 - mu * mu - (log_sigma * 2.0).exp()).sum(axis=1).mean() * (-0.5 / n)
            loss = loss + kl
        return loss

    def parameters(self) -> list[Tensor]:
        return [self.weight1, self.weight_mu] + ([self.weight_sigma] if self.variational else [])


def auc_and_ap(positive_scores: list[float], negative_scores: list[float]) -> tuple[float, float]:
    """Area under the ROC curve, and average precision (both as sklearn computes them)."""

    ranked = sorted([(s, 1) for s in positive_scores] + [(s, 0) for s in negative_scores], key=lambda p: -p[0])
    positives, negatives = len(positive_scores), len(negative_scores)
    # AUC: the chance that a random positive outscores a random negative (ties count half).
    wins = 0.0
    for p in positive_scores:
        for q in negative_scores:
            wins += 1.0 if p > q else (0.5 if p == q else 0.0)
    auc = wins / (positives * negatives)
    # AP: precision at each positive, averaged (scores assumed distinct).
    hits, total = 0, 0.0
    for rank, (_, label) in enumerate(ranked, 1):
        if label:
            hits += 1
            total += hits / rank
    return auc, total / positives


def link_scores(mu: list[float], latent: int, pairs: list[tuple[int, int]]) -> list[float]:
    return [sum(mu[a * latent + k] * mu[b * latent + k] for k in range(latent)) for a, b in pairs]


class NanoGradRunner:
    def __init__(self, model: GraphAutoencoder, graph: CoraLinks, lr: float) -> None:
        self.model, self.graph = model, graph
        self.optimizer = TensorAdam(model.parameters(), lr=lr)

    def train_step(self, noise: list[float] | None) -> float:
        self.optimizer.zero_grad()
        latent = self.model.weight_mu.shape[1]
        loss = self.model.loss(self.graph, None if noise is None else Tensor(noise, (self.graph.nodes, latent)))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def embeddings(self) -> list[float]:
        with no_grad():
            return list(self.model.encode(self.graph)[0].data)

    def weights(self) -> list[list[float]]:
        return [list(p.data) for p in self.model.parameters()]


class TorchRunner:
    def __init__(self, model: GraphAutoencoder, graph: CoraLinks, lr: float) -> None:
        import torch

        self.torch, self.graph, self.variational = torch, graph, model.variational

        def sparse(matrix: SparseMatrix):
            rows = [r for r in range(matrix.shape[0]) for _ in range(matrix.row_start[r], matrix.row_start[r + 1])]
            return torch.sparse_coo_tensor([rows, list(matrix.cols)], list(matrix.values), matrix.shape,
                                           dtype=torch.float64, check_invariants=True).coalesce()

        self.adjacency, self.features = sparse(graph.adjacency), sparse(graph.features)
        self.parameters = [torch.tensor(list(p.data), dtype=torch.float64).reshape(p.shape).requires_grad_()
                           for p in model.parameters()]
        self.optimizer = torch.optim.Adam(self.parameters, lr=lr)
        n = graph.nodes
        self.targets = torch.zeros(n * n, dtype=torch.float64)
        self.targets[torch.tensor(graph.positives)] = 1.0
        positives = len(graph.positives)
        self.pos_weight, self.norm = (n * n - positives) / positives, n * n / (2.0 * (n * n - positives))

    def _encode(self):
        torch = self.torch
        hidden = torch.relu(torch.sparse.mm(self.adjacency, torch.sparse.mm(self.features, self.parameters[0])))
        mu = torch.sparse.mm(self.adjacency, hidden @ self.parameters[1])
        log_sigma = torch.sparse.mm(self.adjacency, hidden @ self.parameters[2]) if self.variational else None
        return mu, log_sigma

    def train_step(self, noise) -> float:
        torch, n = self.torch, self.graph.nodes
        self.optimizer.zero_grad()
        mu, log_sigma = self._encode()
        z = mu + torch.tensor(noise, dtype=torch.float64).reshape(mu.shape) * log_sigma.exp() if self.variational else mu
        logits = (z @ z.T).reshape(-1)
        loss = self.norm * torch.nn.functional.binary_cross_entropy_with_logits(
            logits, self.targets, pos_weight=torch.tensor(self.pos_weight, dtype=torch.float64))
        if self.variational:
            loss = loss - 0.5 / n * (1 + 2 * log_sigma - mu ** 2 - log_sigma.exp() ** 2).sum(1).mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def embeddings(self) -> list[float]:
        with self.torch.no_grad():
            return self._encode()[0].reshape(-1).tolist()

    def weights(self) -> list[list[float]]:
        return [p.detach().reshape(-1).tolist() for p in self.parameters]


def evaluate(runner, graph: CoraLinks, latent: int, split: str) -> tuple[float, float]:
    mu = runner.embeddings()
    positives = graph.test_edges if split == "test" else graph.validation_edges
    negatives = graph.test_non_edges if split == "test" else graph.validation_non_edges
    return auc_and_ap(link_scores(mu, latent, positives), link_scores(mu, latent, negatives))


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="data/cora")
    parser.add_argument("--model", choices=("gae", "vgae"), default="vgae")
    parser.add_argument("--frameworks", nargs="+", choices=("nanograd", "torch"), default=["nanograd"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--latent", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=1, help="edge splits and initializations")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args(argv)

    results = {framework: [] for framework in args.frameworks}
    runners = {}
    for seed in range(args.seeds):
        graph = CoraLinks(args.data, seed)
        if seed == 0:
            print(f"{graph.nodes} nodes, {graph.edges} edges: {len(graph.train_edges)} to train on, "
                  f"{len(graph.validation_edges)} to validate, {len(graph.test_edges)} to test", flush=True)
        for framework in args.frameworks:
            model = GraphAutoencoder(graph.features.shape[1], hidden=args.hidden, latent=args.latent,
                                     variational=args.model == "vgae", seed=seed)
            if framework == "torch":
                import torch

                torch.set_num_threads(1)
                runners[framework] = runner = TorchRunner(model, graph, args.lr)
            else:
                runners[framework] = runner = NanoGradRunner(model, graph, args.lr)
            rng, start = random.Random(seed + 100), perf_counter()
            for epoch in range(1, args.epochs + 1):
                noise = [rng.gauss(0, 1) for _ in range(graph.nodes * args.latent)] if args.model == "vgae" else None
                loss = runner.train_step(noise)
                if epoch % args.log_every == 0:
                    auc, ap = evaluate(runner, graph, args.latent, "validation")
                    print(f"  {framework} seed {seed} epoch {epoch:3d}: loss {loss:.4f}, "
                          f"validation AUC {100 * auc:.1f}, AP {100 * ap:.1f}", flush=True)
            auc, ap = evaluate(runner, graph, args.latent, "test")
            results[framework].append((auc, ap, perf_counter() - start))
            print(f"{framework:8s} {args.model} seed {seed}: test AUC {100 * auc:.1f}, AP {100 * ap:.1f} "
                  f"({perf_counter() - start:.0f} s)", flush=True)
    summary = {}
    for framework, rows in results.items():
        mean_auc = sum(r[0] for r in rows) / len(rows)
        mean_ap = sum(r[1] for r in rows) / len(rows)
        summary[framework] = {"auc": mean_auc, "ap": mean_ap, "seconds": sum(r[2] for r in rows) / len(rows)}
        print(f"{framework:8s} {args.model}: test AUC {100 * mean_auc:.1f}, AP {100 * mean_ap:.1f} "
              f"over {len(rows)} seeds", flush=True)
    if len(runners) == 2:
        difference = max(abs(a - b) for x, y in zip(runners["nanograd"].weights(), runners["torch"].weights())
                         for a, b in zip(x, y))
        print(f"largest weight difference after training (last seed): {difference:.1e}")
        summary["max_weight_difference"] = difference
    return summary


if __name__ == "__main__":
    main()
