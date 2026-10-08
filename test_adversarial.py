import contextlib
import io
import random
import unittest

import adversarial
import kernels


class AdversarialTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = random.Random(0)
        self.images = [rng.random() for _ in range(4 * adversarial.PIXELS)]
        self.labels = [3, 1, 4, 1]
        self.runner = adversarial.NanoGradRunner(adversarial.small_cnn(seed=1), 1e-3)

    def test_input_gradient_matches_finite_differences(self) -> None:
        grad, _ = self.runner.input_gradient(self.images, self.labels)
        eps = 1e-6
        for index in (5, 400, 1000, 3000):
            up, down = list(self.images), list(self.images)
            up[index] += eps
            down[index] -= eps

            def loss(x):
                return sum(adversarial.row_losses(self.runner.logits(x), self.labels)) / len(self.labels)

            self.assertAlmostEqual(grad[index], (loss(up) - loss(down)) / (2 * eps), places=6)

    def test_attacks_stay_in_the_box_and_raise_the_loss(self) -> None:
        eps = 0.1
        clean = sum(adversarial.row_losses(self.runner.logits(self.images), self.labels))
        for delta in (adversarial.fgsm(self.runner, self.images, self.labels, eps),
                      adversarial.pgd(self.runner, self.images, self.labels, eps=eps, alpha=0.02, steps=10,
                                      restarts=2, rng=random.Random(3))):
            attacked = kernels.add(self.images, delta)
            self.assertTrue(all(-eps - 1e-12 <= d <= eps + 1e-12 for d in delta))
            self.assertTrue(all(-1e-12 <= v <= 1 + 1e-12 for v in attacked))
            self.assertGreater(sum(adversarial.row_losses(self.runner.logits(attacked), self.labels)), clean)

    def test_matches_torch(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")
        with contextlib.redirect_stdout(io.StringIO()):
            result = adversarial.main(["--frameworks", "nanograd", "torch", "--epochs", "1", "--train-limit", "300",
                                       "--test-limit", "50", "--pgd-steps", "5", "--restarts", "2"])
        for name in ("clean", "fgsm", "pgd"):
            self.assertEqual(result["nanograd"][name], result["torch"][name])
        self.assertLess(result["max_weight_difference"], 1e-10)


if __name__ == "__main__":
    unittest.main()
