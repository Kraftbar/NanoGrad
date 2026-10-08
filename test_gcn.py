import contextlib
import importlib.util
import io
import random
import tempfile
import unittest
from pathlib import Path

from gcn import Graph, main
from tensor import SparseMatrix, Tensor, matmul, sparse_matmul
from test_gradcheck import max_gradient_error


HAS_TORCH = importlib.util.find_spec("torch") is not None


def tiny_cora(folder: Path, nodes: int = 1600) -> None:
    """A random graph in Cora's file format: two topics, linked mostly within a topic."""

    rng = random.Random(0)
    topics = [node % 2 for node in range(nodes)]
    lines = []
    for node in range(nodes):
        words = ["1" if rng.random() < (0.3 if (w % 2) == topics[node] else 0.1) else "0" for w in range(6)]
        words[0] = "1"                                   # every paper uses at least one word
        lines.append(f"p{node} " + " ".join(words) + f" topic{topics[node]}")
    (folder / "cora.content").write_text("\n".join(lines) + "\n")
    links = [(a, b) for a in range(nodes) for b in [rng.randrange(nodes)] if topics[a] == topics[b] and a != b]
    (folder / "cora.cites").write_text("\n".join(f"p{a} p{b}" for a, b in links) + "\n")


class SparseTests(unittest.TestCase):
    def test_sparse_matmul_matches_dense_and_gradients(self) -> None:
        rng = random.Random(1)
        matrix = SparseMatrix((5, 4), [(rng.randrange(5), rng.randrange(4), rng.uniform(-1, 1)) for _ in range(9)])
        x = Tensor([rng.uniform(-1, 1) for _ in range(12)], (4, 3), requires_grad=True)
        dense = Tensor(matrix.to_dense(), (5, 4))
        for a, b in zip(sparse_matmul(matrix, x).data, matmul(dense, x).data):
            self.assertAlmostEqual(a, b, places=14)
        weights = Tensor([rng.uniform(-1, 1) for _ in range(15)], (5, 3))
        self.assertLess(max_gradient_error(lambda: (sparse_matmul(matrix, x) * weights).sum(), [x]), 1e-8)


class GCNTests(unittest.TestCase):
    def test_graph_rows_average_their_neighbourhood(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tiny_cora(Path(folder))
            graph = Graph(folder)
        dense = graph.adjacency.to_dense()
        size = len(graph.labels)
        for node in (0, 7):
            row = dense[node * size : (node + 1) * size]
            self.assertAlmostEqual(sum(row), 1.0, places=12)
            self.assertGreater(row[node], 0.0)                       # self-loop
        self.assertEqual(dense[3 * size + 9] > 0, dense[9 * size + 3] > 0)   # symmetric links

    def test_training_learns_the_topics(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tiny_cora(Path(folder))
            with contextlib.redirect_stdout(io.StringIO()):
                summary = main(["--data", folder, "--seeds", "1", "--epochs", "60"])
        self.assertGreater(summary["nanograd"]["mean"], 0.7)        # chance is 0.5

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
    def test_trains_identically_in_pytorch(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tiny_cora(Path(folder))
            with contextlib.redirect_stdout(io.StringIO()):
                results = main(["--data", folder, "--compare", "--epochs", "20"])
        self.assertAlmostEqual(results["nanograd"]["loss"], results["torch"]["loss"], places=10)


if __name__ == "__main__":
    unittest.main()
