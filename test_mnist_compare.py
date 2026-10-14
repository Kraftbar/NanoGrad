import importlib.util
import random
import unittest

import kernels
from mnist_compare import IMAGE_SIZE, augmented_pixels, NanoGradRunner, TorchRunner, build_nanograd, epoch_batches


class EpochBatchTests(unittest.TestCase):
    def test_batches_cover_every_example_once(self) -> None:
        batches = epoch_batches(10, 4, random.Random(0))
        self.assertEqual([len(batch) for batch in batches], [4, 4, 2])
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(10)))


@unittest.skipIf(kernels.BACKEND == "cuda", "worker processes are forked, and CUDA does not survive a fork")
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



class AugmentTests(unittest.TestCase):
    def test_shift_and_flip_move_pixels_and_fill_zeros(self) -> None:
        image = bytes(range(1, 17))                        # one 1 x 4 x 4 image, values 1..16
        shapes = (1, 4, 4)
        seen_shift = seen_mirror = False
        for seed in range(2000):       # an unshifted mirror is a 1-in-162 event
            out = augmented_pixels([image], [0], shapes, random.Random(seed), flip=True)
            self.assertEqual(len(out), 16)
            values = {round(v * 255) for v in out} - {0}
            self.assertTrue(values <= set(range(1, 17)))
            original = [v / 255 for v in image]
            if out != original:
                seen_shift = True
            if out == [v / 255 for row in range(4) for v in reversed(image[row * 4 : row * 4 + 4])]:
                seen_mirror = True
        self.assertTrue(seen_shift)
        self.assertTrue(seen_mirror)


if __name__ == "__main__":
    unittest.main()
