import random
import unittest

import fewshot


class FewShotTest(unittest.TestCase):
    def test_rotate_and_shrink(self) -> None:
        image = [[1, 2], [3, 4]]
        self.assertEqual(fewshot.rotate(image, 90), [[2, 4], [1, 3]])       # counter-clockwise
        self.assertEqual(fewshot.rotate(image, 360), image)
        page = [[255] * 56 for _ in range(56)]
        page[1][1] = 0                    # ink where output pixel (0, 0) samples: (0 + 0.5) * 56 / 28 = 1
        small = fewshot.shrink(page)
        self.assertEqual((len(small), small[0], sum(small)), (28 * 28, 1, 1))

    def test_scores_are_negative_squared_distances_to_prototypes(self) -> None:
        runner = fewshot.NanoGradRunner(fewshot.encoder(channels=4), 1e-3)
        rng = random.Random(1)
        classes = {f"c{n}": [[rng.randrange(2) for _ in range(28 * 28)] for _ in range(4)] for n in range(3)}
        support, queries, way = fewshot.episode_batch(classes, 3, 2, 1, rng)
        runner.model.eval()
        scores = runner._scores(support, queries, way).data
        vectors = runner.embed(support + queries)
        prototypes = [[(a + b) / 2 for a, b in zip(vectors[2 * n], vectors[2 * n + 1])] for n in range(3)]
        for q in range(3):
            for n in range(3):
                distance = sum((a - b) ** 2 for a, b in zip(vectors[6 + q], prototypes[n]))
                self.assertAlmostEqual(scores[q * 3 + n], -distance, places=9)

    def test_training_matches_torch(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")
        rng = random.Random(2)
        classes = {f"c{n}": [[rng.randrange(2) for _ in range(28 * 28)] for _ in range(4)] for n in range(5)}
        model = fewshot.encoder(channels=8)
        runners = [fewshot.NanoGradRunner(model, 1e-3), fewshot.TorchRunner(model, 1e-3)]
        for step in range(3):
            batch = fewshot.episode_batch(classes, 4, 1, 2, random.Random(step))
            losses = [runner.train_step(*batch, 1e-3)[0] for runner in runners]
            self.assertAlmostEqual(losses[0], losses[1], places=9)
        difference = max(abs(a - b) for x, y in zip(runners[0].weights(), runners[1].weights()) for a, b in zip(x, y))
        self.assertLess(difference, 1e-10)


if __name__ == "__main__":
    unittest.main()
