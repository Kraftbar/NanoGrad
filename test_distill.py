import contextlib
import io
import random
import unittest

import distill
from tensor import Tensor


class DistillTest(unittest.TestCase):
    def test_softened_targets(self) -> None:
        q = distill.softened([2.0, 0.0] + [0.0] * 8, 1.0)
        self.assertAlmostEqual(sum(q), 1.0)
        hot = distill.softened([20.0] + [0.0] * 9, 1.0)
        warm = distill.softened([20.0] + [0.0] * 9, 20.0)
        self.assertGreater(hot[0], 0.99)
        self.assertLess(warm[0], 0.3)                  # a high temperature shows the other classes

    def test_jitter_shifts_the_image(self) -> None:
        image = bytes(255 if (y, x) == (10, 10) else 0 for y in range(28) for x in range(28))
        shifted = distill.jitter(image, random.Random(3))
        (spot,) = [i for i, v in enumerate(shifted) if v]
        self.assertLessEqual(abs(spot // 28 - 10), 2)
        self.assertLessEqual(abs(spot % 28 - 10), 2)

    def test_loss_without_soft_targets_is_cross_entropy(self) -> None:
        logits = Tensor([0.5, -0.2] + [0.0] * 8, (1, 10))
        plain = distill.softmax_cross_entropy(logits, [3]).item()
        self.assertEqual(distill.distillation_loss(logits, None, [3], temperature=20, alpha=0.9).item(), plain)

    def test_matches_torch(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")
        with contextlib.redirect_stdout(io.StringIO()):
            result = distill.main(["--compare", "--train-limit", "500"])
        self.assertLess(result["max_weight_difference"], 1e-10)


if __name__ == "__main__":
    unittest.main()
