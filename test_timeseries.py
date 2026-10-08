import importlib.util
import random
import unittest

from tensor import Tensor, matmul
from timeseries import DLinear, moving_average_matrix, splits, MONTH


HAS_TORCH = importlib.util.find_spec("torch") is not None


class TimeSeriesTests(unittest.TestCase):
    def test_moving_average_repeats_the_ends(self) -> None:
        a = moving_average_matrix(5, 3)
        series = [1.0, 2.0, 3.0, 4.0, 10.0]
        trend = [sum(a[t * 5 + j] * series[j] for j in range(5)) for t in range(5)]
        self.assertAlmostEqual(trend[0], (1 + 1 + 2) / 3)          # padded with the first value
        self.assertAlmostEqual(trend[2], 3.0)
        self.assertAlmostEqual(trend[4], (4 + 10 + 10) / 3)

    def test_splits_follow_the_benchmark(self) -> None:
        ranges = splits([[0.0]] * (20 * MONTH), 336, 96)
        self.assertEqual(ranges["train"], (0, 12 * MONTH))
        self.assertEqual(ranges["test"], (16 * MONTH - 336, 20 * MONTH))

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
    def test_matches_the_authors_formulation(self) -> None:
        import torch

        from timeseries import TorchRunner

        model = DLinear(30, 6, window=5, seed=1)
        x = [random.Random(2).uniform(-1, 1) for _ in range(2 * 3 * 30)]
        ours = model(Tensor(x, (2, 3, 30)))
        twin = TorchRunner(model, lr=0.01, window=5)
        theirs = twin._forward(torch.tensor(x, dtype=torch.float64).reshape(2, 3, 30))
        for a, b in zip(ours.data, theirs.reshape(-1).tolist()):
            self.assertAlmostEqual(a, b, places=12)


if __name__ == "__main__":
    unittest.main()
