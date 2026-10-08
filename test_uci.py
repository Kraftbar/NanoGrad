import contextlib
import io
import math
import random
import tempfile
import unittest
from pathlib import Path

import uci


def write_toy_dataset(folder: Path, rows: int = 120, seed: int = 0) -> None:
    """y = 3 x0 - 2 x1 + noise, in the benchmark's file layout, with 2 splits."""

    rng = random.Random(seed)
    lines = []
    for _ in range(rows):
        x0, x1 = rng.uniform(-1, 1), rng.uniform(-1, 1)
        lines.append(f"{x0} {x1} {3 * x0 - 2 * x1 + rng.gauss(0, 0.1)}")
    (folder / "data.txt").write_text("\n".join(lines) + "\n")
    for name, value in (("index_features", "0\n1"), ("index_target", "2"), ("n_splits", "2"), ("n_hidden", "20"),
                        ("n_epochs", "30"), ("dropout_rates", "0.01\n0.05"), ("tau_values", "10\n50")):
        (folder / f"{name}.txt").write_text(value + "\n")
    for split in range(2):
        order = list(range(rows))
        rng.shuffle(order)
        (folder / f"index_train_{split}.txt").write_text("\n".join(map(str, order[:100])))
        (folder / f"index_test_{split}.txt").write_text("\n".join(map(str, order[100:])))


class UCITest(unittest.TestCase):
    def test_standardizer(self) -> None:
        scale = uci.Standardizer([[1.0, 5.0], [3.0, 5.0]])
        self.assertEqual(scale([[1.0, 5.0], [3.0, 5.0]]), [[-1.0, 0.0], [1.0, 0.0]])   # constant column: std 1

    def test_mc_log_likelihood_of_a_fixed_prediction(self) -> None:
        class Constant:                     # every sample predicts 0 (in standardized units)
            def sample(self, x, copies):
                return [0.0] * (len(x) * copies)

        scale = uci.Standardizer([[0.0], [2.0]])
        rmse, ll = uci.evaluate(Constant(), scale, (1.0, 2.0), [[0.0]], [2.0], tau=4.0, samples=7)
        self.assertAlmostEqual(rmse, 1.0)
        self.assertAlmostEqual(ll, -0.5 * 4.0 - 0.5 * math.log(2 * math.pi) + 0.5 * math.log(4.0))

    def test_learns_a_linear_function(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            name = Path(folder) / "toy"
            name.mkdir()
            write_toy_dataset(name)
            with contextlib.redirect_stdout(io.StringIO()):
                results = uci.main(["--data", folder, "--dataset", "toy", "--epoch-multiplier", "30",
                                    "--samples", "200", "--search-samples", "50"])
        self.assertLess(results[("toy", "nanograd")]["rmse"][0], 0.3)    # noise 0.1; the target spread is about 2


if __name__ == "__main__":
    unittest.main()
