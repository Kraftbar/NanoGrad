import random
import unittest

import segment
from losses import softmax_cross_entropy
from tensor import Tensor


class SegmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.saved = segment.HEIGHT, segment.WIDTH
        segment.HEIGHT, segment.WIDTH = 8, 16           # three poolings still divide evenly

    def tearDown(self) -> None:
        segment.HEIGHT, segment.WIDTH = self.saved

    def pairs(self, count: int, seed: int = 0):
        rng = random.Random(seed)
        plane = segment.HEIGHT * segment.WIDTH
        return [(bytes(rng.randrange(256) for _ in range(3 * plane)),
                 bytes(rng.choice([0, 3, 8, segment.VOID]) for _ in range(plane))) for _ in range(count)]

    def test_pixel_loss_skips_void(self) -> None:
        rng = random.Random(1)
        scores = Tensor([rng.uniform(-1, 1) for _ in range(2 * 11 * 2 * 2)], (2, 11, 2, 2))
        labels = [0, segment.VOID, 5, 10, segment.VOID, 1, 2, 3]
        rows, targets = [], []
        for b in range(2):
            for pixel in range(4):
                label = labels[b * 4 + pixel]
                if label != segment.VOID:
                    rows += [scores.data[b * 44 + c * 4 + pixel] for c in range(11)]
                    targets.append(label)
        expected = softmax_cross_entropy(Tensor(rows, (len(targets), 11)), targets).item()
        self.assertAlmostEqual(segment.pixel_loss(scores, labels).item(), expected, places=12)

    def test_flip_mirrors_rows(self) -> None:
        (pair,) = self.pairs(1)
        stats = [(0.0, 1.0)] * 3
        straight, labels = segment.batch_arrays([pair], stats, [False])
        flipped, flipped_labels = segment.batch_arrays([pair], stats, [True])
        w = segment.WIDTH
        self.assertEqual(flipped[:w], straight[:w][::-1])
        self.assertEqual(flipped_labels[w : 2 * w], labels[w : 2 * w][::-1])

    def test_metrics(self) -> None:
        pairs = self.pairs(2)

        class Oracle:                                    # right everywhere except one class
            def predict(self, pixels, batch):
                labels = [v for _, label in pairs[:batch] for v in label]
                return [0 if v == 8 else v for v in labels]

        scores = segment.evaluate(Oracle(), pairs, [(0.0, 1.0)] * 3, batch=2)
        self.assertLess(scores["global"], 1.0)
        self.assertEqual(scores["iou"]["road"], 1.0)
        self.assertEqual(scores["iou"]["car"], 0.0)

    def test_matches_torch(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")
        pairs = self.pairs(4, seed=2)
        stats = segment.channel_stats(pairs)
        model = segment.UNet(seed=3)
        weights = segment.median_frequency_weights(pairs)
        runners = [segment.NanoGradRunner(model, 1e-3, weights), segment.TorchRunner(model, 1e-3, weights)]
        for step in range(2):
            pixels, labels = segment.batch_arrays(pairs[2 * step : 2 * step + 2], stats, [step == 1, False])
            losses = [runner.train_step(pixels, labels, 2) for runner in runners]
            self.assertAlmostEqual(losses[0], losses[1], places=10)
        difference = max(abs(a - b) for x, y in zip(runners[0].weights(), runners[1].weights()) for a, b in zip(x, y))
        self.assertLess(difference, 1e-10)
        pixels, _ = segment.batch_arrays(pairs[:2], stats, [False, False])
        self.assertEqual(runners[0].predict(pixels, 2), runners[1].predict(pixels, 2))


if __name__ == "__main__":
    unittest.main()
