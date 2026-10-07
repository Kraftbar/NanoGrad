import importlib.util
import random
import unittest

from mnist_compare import IMAGE_SIZE, NanoGradRunner, TorchRunner, build_nanograd, epoch_batches


class EpochBatchTests(unittest.TestCase):
    def test_batches_cover_every_example_once(self) -> None:
        batches = epoch_batches(10, 4, random.Random(0))
        self.assertEqual([len(batch) for batch in batches], [4, 4, 2])
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(10)))


class WorkerTests(unittest.TestCase):
    def test_workers_give_the_single_process_step(self) -> None:
        rng = random.Random(5)
        pixels = [rng.random() for _ in range(6 * IMAGE_SIZE)]
        labels = [rng.randrange(10) for _ in range(6)]
        single = NanoGradRunner(build_nanograd(seed=3), lr=1e-3)
        split = NanoGradRunner(build_nanograd(seed=3), lr=1e-3, workers=2)
        try:
            self.assertAlmostEqual(single.train_step(pixels, labels), split.train_step(pixels, labels), places=12)
        finally:
            split.close()
        for left, right in zip(single.weights(), split.weights()):
            for a, b in zip(left, right):
                self.assertAlmostEqual(a, b, places=12)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class TorchAgreementTests(unittest.TestCase):
    def test_same_losses_and_weights_after_a_few_steps(self) -> None:
        model = build_nanograd(seed=3)
        torch_runner = TorchRunner(model, lr=1e-3, dtype="float64", threads=1)
        nanograd_runner = NanoGradRunner(model, lr=1e-3)
        rng = random.Random(4)
        for _ in range(3):
            pixels = [rng.random() for _ in range(4 * IMAGE_SIZE)]
            labels = [rng.randrange(10) for _ in range(4)]
            self.assertAlmostEqual(
                nanograd_runner.train_step(pixels, labels), torch_runner.train_step(pixels, labels), places=12,
            )
        for left, right in zip(nanograd_runner.weights(), torch_runner.weights()):
            for a, b in zip(left, right):
                self.assertAlmostEqual(a, b, places=12)
        self.assertEqual(nanograd_runner.predict(pixels, 4), torch_runner.predict(pixels, 4))


if __name__ == "__main__":
    unittest.main()
