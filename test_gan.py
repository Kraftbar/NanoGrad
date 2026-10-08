import argparse
import importlib.util
import random
import unittest

from gan import GAN, discriminator, generator
from tensor import Tensor, conv_transpose2d
from tensor_nn import LeakyReLU, Sequential, Tanh
from test_gradcheck import max_gradient_error


HAS_TORCH = importlib.util.find_spec("torch") is not None


def values(count: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(count)]


class GANTests(unittest.TestCase):
    def test_transposed_convolution_grows_the_image_and_has_correct_gradients(self) -> None:
        image = Tensor(values(2 * 3 * 4 * 4, 1), (2, 3, 4, 4), requires_grad=True)
        kernel = Tensor(values(3 * 2 * 4 * 4, 2), (3, 2, 4, 4), requires_grad=True)
        out = conv_transpose2d(image, kernel, stride=2, padding=1)
        self.assertEqual(out.shape, (2, 2, 8, 8))
        weights = Tensor(values(out.numel, 3), out.shape)
        error = max_gradient_error(lambda: (conv_transpose2d(image, kernel, stride=2, padding=1) * weights).sum(),
                                   [image, kernel])
        self.assertLess(error, 1e-8)

    def test_shapes_and_leaky_relu(self) -> None:
        self.assertEqual(generator(16, 8)(Tensor(values(2 * 16, 4), (2, 16, 1, 1))).shape, (2, 1, 28, 28))
        self.assertEqual(discriminator(8)(Tensor(values(2 * 784, 5), (2, 1, 28, 28))).shape, (2, 1))
        out = Sequential(LeakyReLU(0.2), Tanh())(Tensor([-1.0, 2.0], (2,)))
        self.assertAlmostEqual(out.data[0], -0.197375320224904, places=12)       # tanh(-0.2)

    def test_training_step_runs(self) -> None:
        gan = GAN(argparse.Namespace(seed=0, noise=8, width=4, lr=2e-4))
        d, g = gan.train_step(values(4 * 784, 6), values(4 * 8, 7), 4)
        self.assertGreater(d, 0.0)
        self.assertGreater(g, 0.0)

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
    def test_twin_matches_both_networks(self) -> None:
        import torch

        from torch_twin import TorchTwin

        g, d = generator(8, 4, seed=1), discriminator(4, seed=2)
        z = values(3 * 8, 8)
        fake = g(Tensor(z, (3, 8, 1, 1)))
        theirs = TorchTwin(g)(torch.tensor(z, dtype=torch.float64).reshape(3, 8, 1, 1))
        for a, b in zip(fake.data, theirs.reshape(-1).tolist()):
            self.assertAlmostEqual(a, b, places=12)
        scores = d(fake)
        twin_scores = TorchTwin(d)(theirs.detach())
        for a, b in zip(scores.data, twin_scores.reshape(-1).tolist()):
            self.assertAlmostEqual(a, b, places=11)


if __name__ == "__main__":
    unittest.main()
