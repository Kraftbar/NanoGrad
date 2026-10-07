import importlib.util
import random
import unittest

from gradcheck import max_gradient_error
from losses import binary_cross_entropy_with_logits
from tensor import Tensor
from vae import VAE, vae_loss


def random_values(count, seed, low=-1.0, high=1.0):
    rng = random.Random(seed)
    return [rng.uniform(low, high) for _ in range(count)]


class VAETests(unittest.TestCase):
    def test_bce_with_logits_matches_the_definition_and_its_gradient(self) -> None:
        logits = Tensor(random_values(12, 1, -5.0, 5.0), (3, 4), requires_grad=True)
        targets = Tensor(random_values(12, 2, 0.0, 1.0), (3, 4))
        import math

        expected = 0.0
        for x, t in zip(logits.data, targets.data):
            p = 1.0 / (1.0 + math.exp(-x))
            expected -= t * math.log(p) + (1.0 - t) * math.log(1.0 - p)
        self.assertAlmostEqual(binary_cross_entropy_with_logits(logits, targets).item(), expected / 12, places=12)
        error = max_gradient_error(lambda: binary_cross_entropy_with_logits(logits, targets, reduction="sum"), [logits])
        self.assertLess(error, 1e-6)
        huge = Tensor([1000.0, -1000.0], (2,))
        self.assertAlmostEqual(binary_cross_entropy_with_logits(huge, Tensor([1.0, 0.0], (2,))).item(), 0.0)

    def test_vae_gradients(self) -> None:
        model = VAE(pixels=12, hidden=8, latent=3, seed=3)
        images = Tensor(random_values(24, 4, 0.0, 1.0), (2, 12))
        noise = Tensor(random_values(6, 5), (2, 3))

        def loss():
            logits, mean, log_var = model(images, noise)
            return vae_loss(logits, images, mean, log_var)[0]

        self.assertLess(max_gradient_error(loss, model.parameters()), 1e-5)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_trains_like_pytorch(self) -> None:
        from vae_compare import NanoGradRunner, TorchRunner

        model = VAE(pixels=784, hidden=32, latent=4, seed=6)
        theirs = TorchRunner(model, lr=1e-3, threads=1)
        ours = NanoGradRunner(model, lr=1e-3)
        for step in range(3):
            pixels = random_values(5 * 784, 10 + step, 0.0, 1.0)
            noise = random_values(5 * 4, 20 + step)
            self.assertAlmostEqual(ours.train_step(pixels, noise, 5), theirs.train_step(pixels, noise, 5), places=8)
        for left, right in zip(ours.weights(), theirs.weights()):
            for a, b in zip(left, right):
                self.assertAlmostEqual(a, b, places=12)


if __name__ == "__main__":
    unittest.main()
