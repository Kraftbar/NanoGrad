import contextlib
import io
import random
import tempfile
import unittest
from pathlib import Path

import linkpred
from test_gcn import tiny_cora


class LinkPredictionTest(unittest.TestCase):
    def test_auc_and_ap(self) -> None:
        self.assertEqual(linkpred.auc_and_ap([0.9, 0.8], [0.1, 0.2]), (1.0, 1.0))
        auc, ap = linkpred.auc_and_ap([0.9, 0.3], [0.5, 0.1])        # ranking: +, -, +, -
        self.assertAlmostEqual(auc, 0.75)
        self.assertAlmostEqual(ap, (1 / 1 + 2 / 3) / 2)

    def test_split_is_disjoint_and_encoder_sees_only_training_edges(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tiny_cora(Path(folder), nodes=300)
            graph = linkpred.CoraLinks(folder, seed=0)
        held_out = set(graph.test_edges) | set(graph.validation_edges)
        self.assertFalse(held_out & set(graph.train_edges))
        known = held_out | set(graph.train_edges)
        self.assertFalse(set(graph.test_non_edges) & known)
        n = graph.nodes
        for a, b in held_out:
            self.assertNotIn(a * n + b, graph.positives)
        for a, b in graph.train_edges:
            self.assertIn(a * n + b, graph.positives)
            self.assertIn(b * n + a, graph.positives)

    def test_matches_torch(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")
        with tempfile.TemporaryDirectory() as folder:
            tiny_cora(Path(folder), nodes=300)
            for model in ("gae", "vgae"):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = linkpred.main(["--data", folder, "--model", model, "--epochs", "15",
                                            "--frameworks", "nanograd", "torch"])
                self.assertAlmostEqual(result["nanograd"]["auc"], result["torch"]["auc"])
                self.assertLess(result["max_weight_difference"], 1e-10)


if __name__ == "__main__":
    unittest.main()
