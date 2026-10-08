import importlib.util
import random
import unittest

from mobilenet import MobileNet
from tensor import Tensor, depthwise_conv2d
from test_gradcheck import max_gradient_error


HAS_TORCH = importlib.util.find_spec("torch") is not None


def values(count: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(count)]


class DepthwiseTests(unittest.TestCase):
    def test_each_channel_uses_only_its_own_kernel(self) -> None:
        image = Tensor(values(2 * 2 * 4 * 4, 1), (2, 2, 4, 4))
        kernel = Tensor([0.0] * 9 + [1.0] * 9, (2, 1, 3, 3))          # channel 0 off, channel 1 sums
        out = depthwise_conv2d(image, kernel)
        self.assertEqual(out.shape, (2, 2, 2, 2))
        self.assertTrue(all(value == 0.0 for value in out.data[:4]))
        expected = sum(image.data[16 + r * 4 + c] for r in range(3) for c in range(3))
        self.assertAlmostEqual(out.data[4], expected)

    def test_gradients(self) -> None:
        image = Tensor(values(2 * 3 * 6 * 6, 2), (2, 3, 6, 6), requires_grad=True)
        kernel = Tensor(values(3 * 9, 3), (3, 1, 3, 3), requires_grad=True)
        weights = Tensor(values(2 * 3 * 3 * 3, 4), (2, 3, 3, 3))
        error = max_gradient_error(lambda: (depthwise_conv2d(image, kernel, stride=2, padding=1) * weights).sum(),
                                   [image, kernel])
        self.assertLess(error, 1e-8)

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
    def test_mobilenet_matches_pytorch(self) -> None:
        import torch

        from torch_twin import TorchTwin

        model = MobileNet(image_shape=(3, 8, 8), stem=4, seed=1)
        images = values(2 * 3 * 8 * 8, 5)
        ours = model(Tensor(images, (2, 3, 8, 8)))
        theirs = TorchTwin(model)(torch.tensor(images, dtype=torch.float64).reshape(2, 3, 8, 8))
        for a, b in zip(ours.data, theirs.reshape(-1).tolist()):
            self.assertAlmostEqual(a, b, places=10)


if __name__ == "__main__":
    unittest.main()
