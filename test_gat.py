import contextlib
import io
import random
import tempfile
import unittest
from pathlib import Path

from gat import GraphAttention, edge_list, elu, main
from gcn import Graph
from tensor import Tensor
from test_gcn import tiny_cora
from test_gradcheck import max_gradient_error


class GATTests(unittest.TestCase):
    def test_elu(self) -> None:
        out = elu(Tensor([-1.0, 0.0, 2.0], (3,)))
        self.assertAlmostEqual(out.data[0], 2.718281828459045 ** -1 - 1.0, places=12)
        self.assertEqual(list(out.data)[1:], [0.0, 2.0])

    def test_attention_averages_neighbours(self) -> None:
        """With one feature and z = h, each output is a weighted mean of the
        neighbours' values: it must lie between their smallest and largest."""

        rng = random.Random(0)
        layer = GraphAttention(1, 1, heads=1, dropout=0.0, rng=rng)
        layer.weight.data = [1.0]
        sources, targets = [0, 1, 2, 1, 2], [0, 0, 0, 1, 2]
        values = [0.5, -1.0, 2.0]
        out = layer(Tensor(values, (3, 1)), sources, targets, 3)
        self.assertGreater(out.data[0], -1.0)
        self.assertLess(out.data[0], 2.0)
        self.assertAlmostEqual(out.data[1], -1.0)                 # only itself

    def test_gradients(self) -> None:
        rng = random.Random(1)
        layer = GraphAttention(3, 2, heads=2, dropout=0.0, rng=rng)
        h = Tensor([rng.uniform(-1, 1) for _ in range(12)], (4, 3), requires_grad=True)
        sources, targets = [0, 1, 2, 3, 1, 3, 0], [0, 0, 1, 2, 3, 3, 3]
        weights = Tensor([rng.uniform(-1, 1) for _ in range(16)], (4, 4))
        error = max_gradient_error(lambda: (layer(h, sources, targets, 4) * weights).sum(),
                                   [h, layer.weight, layer.attend_source, layer.attend_target])
        self.assertLess(error, 1e-8)

    def test_learns_the_toy_graph(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tiny_cora(Path(folder))
            graph = Graph(folder)
            self.assertEqual(len(edge_list(graph)[0]), graph.adjacency.nnz)
            with contextlib.redirect_stdout(io.StringIO()):
                summary = main(["--data", folder, "--seeds", "1", "--epochs", "40", "--heads", "2", "--hidden", "4",
                                     "--dropout", "0.2"])
        self.assertGreater(summary["mean"], 0.7)


if __name__ == "__main__":
    unittest.main()
