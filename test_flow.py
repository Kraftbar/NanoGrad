import contextlib
import io
import math
import random
import unittest

import flow
from tensor import Tensor, no_grad


def determinant(matrix: list[list[float]]) -> float:
    """Gaussian elimination with partial pivoting."""

    a = [row[:] for row in matrix]
    n, det = len(a), 1.0
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if pivot != col:
            a[col], a[pivot], det = a[pivot], a[col], -det
        det *= a[col][col]
        for r in range(col + 1, n):
            factor = a[r][col] / a[col][col]
            for c in range(col, n):
                a[r][c] -= factor * a[col][c]
    return det


class FlowTest(unittest.TestCase):
    def test_made_is_autoregressive(self) -> None:
        dim = 6
        order = [3, 1, 6, 2, 5, 4]                 # each input's position in the ordering
        made = flow.MADE(dim, 16, order, random.Random(0))
        x = [random.Random(1).gauss(0, 1) for _ in range(dim)]
        with no_grad():
            base, _ = made(Tensor(x, (1, dim)))
            for j in range(dim):
                moved = list(x)
                moved[j] += 1.0
                out, _ = made(Tensor(moved, (1, dim)))
                for d in range(dim):
                    changed = abs(out.data[d] - base.data[d]) > 1e-12
                    # u_d depends on x_d itself and on inputs earlier in the order, never later ones.
                    if order[j] > order[d]:
                        self.assertFalse(changed, f"u_{d} depends on later input {j}")

    def test_log_determinant_matches_the_jacobian(self) -> None:
        dim, eps = 4, 1e-6
        model = flow.MAF(dim, 8, 3, seed=2)
        rows = [[random.Random(r).gauss(0, 1) for _ in range(dim)] for r in range(5)]
        model.train()
        with no_grad():                              # set the batch norm statistics, then freeze them
            for norm in model.steps:
                if isinstance(norm, flow.BatchNormFlow):
                    norm.momentum = 1.0
            model.log_prob(Tensor([v for row in rows for v in row], (5, dim)))
        model.eval()
        x = rows[0]
        with no_grad():
            _, log_det = model.transform(Tensor(x, (1, dim)))
            jacobian = [[0.0] * dim for _ in range(dim)]
            for j in range(dim):
                up, down = list(x), list(x)
                up[j] += eps
                down[j] -= eps
                u_up, _ = model.transform(Tensor(up, (1, dim)))
                u_down, _ = model.transform(Tensor(down, (1, dim)))
                for i in range(dim):
                    jacobian[i][j] = (u_up.data[i] - u_down.data[i]) / (2 * eps)
        self.assertAlmostEqual(log_det.data[0], math.log(abs(determinant(jacobian))), places=5)

    def test_logit_jacobian_converts_densities(self) -> None:
        # A uniform density on pixel space [0, 256) is 1/256 per dimension: 8 bits.
        values, jacobians = flow.to_logit([bytes([0, 128, 255])], random.Random(0))
        s = [1 / (1 + math.exp(-v)) for v in values]
        density_in_logit_space = sum(math.log(si * (1 - si)) for si in s) - 3 * math.log(1 - 2 * flow.LAMBDA)
        bits = -(density_in_logit_space + jacobians[0]) / 3 / math.log(2)
        self.assertAlmostEqual(bits, 8.0, places=6)

    def test_matches_torch(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")
        with contextlib.redirect_stdout(io.StringIO()):
            result = flow.main(["--frameworks", "nanograd", "torch", "--epochs", "1", "--train-limit", "300",
                                "--test-limit", "100", "--hidden", "32", "--layers", "2", "--calibration", "200"])
        self.assertAlmostEqual(result["nanograd"]["bits"], result["torch"]["bits"], places=8)
        self.assertLess(result["max_weight_difference"], 1e-8)


if __name__ == "__main__":
    unittest.main()
