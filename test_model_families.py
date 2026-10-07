import importlib.util
import random
import unittest

from losses import softmax_cross_entropy
from mnist_compare import IMAGE_SIZE, NanoGradRunner, build_nanograd
from resnet import SmallResNet
from tensor import Tensor, conv2d, no_grad, pad2d
from tensor_nn import TensorBatchNorm2d
from train import evaluation_mode
from vit import TinyViT


def random_batch(count, seed):
    rng = random.Random(seed)
    return [rng.random() for _ in range(count * IMAGE_SIZE)], [rng.randrange(10) for _ in range(count)]


class ShapeTests(unittest.TestCase):
    def test_padding_and_stride_shapes(self) -> None:
        image = Tensor([1.0] * (2 * 3 * 7 * 7), (2, 3, 7, 7))
        kernel = Tensor([0.1] * (4 * 3 * 3 * 3), (4, 3, 3, 3))
        self.assertEqual(pad2d(image, 1).shape, (2, 3, 9, 9))
        self.assertEqual(conv2d(image, kernel, padding=1).shape, (2, 4, 7, 7))
        self.assertEqual(conv2d(image, kernel, stride=2, padding=1).shape, (2, 4, 4, 4))

    def test_pad2d_puts_zeros_around_the_image(self) -> None:
        image = Tensor([1.0, 2.0, 3.0, 4.0], (2, 2))
        self.assertEqual(pad2d(image, 1).tolist(), [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 3.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])

    def test_models_classify_a_batch(self) -> None:
        pixels, _ = random_batch(3, seed=1)
        images = Tensor(pixels, (3, 1, 28, 28))
        for model in (SmallResNet(), TinyViT()):
            with self.subTest(model=type(model).__name__):
                self.assertEqual(model(images).shape, (3, 10))


class BatchNormTests(unittest.TestCase):
    def test_training_mode_normalizes_each_channel(self) -> None:
        rng = random.Random(2)
        image = Tensor([rng.uniform(-3.0, 5.0) for _ in range(4 * 2 * 3 * 3)], (4, 2, 3, 3))
        out = TensorBatchNorm2d(2)(image)
        for channel in range(2):
            values = [out[b, channel, r, c] for b in range(4) for r in range(3) for c in range(3)]
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            self.assertAlmostEqual(mean, 0.0, places=12)
            self.assertAlmostEqual(variance, 1.0, places=4)

    def test_running_statistics_round_trip(self) -> None:
        model = SmallResNet(seed=3)
        pixels, labels = random_batch(4, seed=4)
        softmax_cross_entropy(model(Tensor(pixels, (4, 1, 28, 28))), labels).backward()
        copy = SmallResNet(seed=99)
        copy.load_state_dict(model.state_dict())
        images = Tensor(pixels, (4, 1, 28, 28))
        with no_grad(), evaluation_mode(model), evaluation_mode(copy):
            self.assertEqual(list(model(images).data), list(copy(images).data))

    def test_vit_state_round_trip(self) -> None:
        model, copy = TinyViT(seed=5), TinyViT(seed=6)
        copy.load_state_dict(model.state_dict())
        pixels, _ = random_batch(2, seed=7)
        images = Tensor(pixels, (2, 1, 28, 28))
        self.assertEqual(list(model(images).data), list(copy(images).data))


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class TorchAgreementTests(unittest.TestCase):
    def test_every_model_trains_like_pytorch(self) -> None:
        from mnist_compare import TorchRunner

        for name in ("lenet", "resnet", "vit"):
            with self.subTest(model=name):
                model = build_nanograd(3, name)
                torch_runner = TorchRunner(model, lr=1e-3, dtype="float64", threads=1)
                ours = NanoGradRunner(model, lr=1e-3)
                for step in range(3):
                    pixels, labels = random_batch(4, seed=10 + step)
                    self.assertAlmostEqual(ours.train_step(pixels, labels), torch_runner.train_step(pixels, labels), places=10)
                for left, right in zip(ours.weights(), torch_runner.weights()):
                    for a, b in zip(left, right):
                        self.assertAlmostEqual(a, b, places=10)
                # Evaluation mode, including batch-norm running statistics.
                pixels, _ = random_batch(5, seed=20)
                self.assertEqual(ours.predict(pixels, 5), torch_runner.predict(pixels, 5))
                ours_running = [
                    values
                    for module in model.modules() if isinstance(module, TensorBatchNorm2d)
                    for values in (module.running_mean, module.running_var)
                ]
                for left, right in zip(ours_running, torch_runner.twin.running_statistics()):
                    for a, b in zip(left, right):
                        self.assertAlmostEqual(a, b, places=10)


if __name__ == "__main__":
    unittest.main()
