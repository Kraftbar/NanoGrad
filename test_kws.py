import math
import random
import unittest

import kws


class KeywordSpottingTest(unittest.TestCase):
    def test_mfcc_shape_and_dct(self) -> None:
        mfcc = kws.MFCC()
        rng = random.Random(0)
        clip = [rng.gauss(0, 0.1) for _ in range(kws.RATE)]
        (features,) = mfcc([clip])
        self.assertEqual(len(features), kws.FRAMES * kws.COEFFICIENTS)
        # The DCT matrix is orthonormal: its columns have unit length and are orthogonal.
        d = mfcc.dct.data
        for a in range(kws.COEFFICIENTS):
            for b in range(kws.COEFFICIENTS):
                dot = sum(d[m * kws.COEFFICIENTS + a] * d[m * kws.COEFFICIENTS + b] for m in range(kws.MELS))
                self.assertAlmostEqual(dot, 1.0 if a == b else 0.0, places=10)

    def test_louder_means_a_larger_first_coefficient(self) -> None:
        mfcc = kws.MFCC()
        tone = [math.sin(2 * math.pi * 440 * n / kws.RATE) for n in range(kws.RATE)]
        quiet, loud = mfcc([[0.01 * v for v in tone], [0.5 * v for v in tone]])
        self.assertGreater(loud[0], quiet[0])          # c0 is the overall log energy (scaled)

    def test_ds_cnn_shapes_and_parameters(self) -> None:
        model = kws.ds_cnn()
        self.assertLess(model.num_parameters(), 30_000)
        runner = kws.NanoGradRunner(model, 1e-3)
        predicted = runner.predict([0.1] * (2 * kws.FRAMES * kws.COEFFICIENTS), 2)
        self.assertEqual(len(predicted), 2)

    def test_matches_torch(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")
        rng = random.Random(1)
        model = kws.ds_cnn(seed=2)
        runners = [kws.NanoGradRunner(model, 1e-3), kws.TorchRunner(model, 1e-3)]
        for _ in range(2):
            features = [rng.gauss(0, 1) for _ in range(8 * kws.FRAMES * kws.COEFFICIENTS)]
            labels = [rng.randrange(12) for _ in range(8)]
            losses = [runner.train_step(features, labels, 1e-3) for runner in runners]
            self.assertAlmostEqual(losses[0], losses[1], places=10)
        difference = max(abs(a - b) for x, y in zip(runners[0].weights(), runners[1].weights()) for a, b in zip(x, y))
        self.assertLess(difference, 1e-10)


if __name__ == "__main__":
    unittest.main()
