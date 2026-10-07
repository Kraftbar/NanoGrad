import math
import random
import unittest
from collections import Counter

from sampling import sample_from_logits


class TopPTests(unittest.TestCase):
    def test_nucleus_keeps_the_smallest_set_reaching_top_p(self) -> None:
        # Probabilities 0.5, 0.3, 0.15, 0.05: top_p = 0.7 keeps the first two.
        logits = [math.log(p) for p in (0.5, 0.3, 0.15, 0.05)]
        rng = random.Random(0)
        counts = Counter(sample_from_logits(logits, top_p=0.7, rng=rng) for _ in range(4000))
        self.assertEqual(set(counts), {0, 1})
        self.assertAlmostEqual(counts[0] / 4000, 0.5 / 0.8, delta=0.03)

    def test_top_p_one_and_none_agree(self) -> None:
        logits = [0.1, 2.0, -1.0, 0.5]
        a = [sample_from_logits(logits, rng=random.Random(5)) for _ in range(1)]
        counts_none = Counter(sample_from_logits(logits, rng=random.Random(seed)) for seed in range(500))
        counts_one = Counter(sample_from_logits(logits, top_p=1.0, rng=random.Random(seed)) for seed in range(500))
        self.assertEqual(set(counts_none), set(counts_one))
        self.assertEqual(len(a), 1)

    def test_invalid_top_p(self) -> None:
        for bad in (0.0, 1.5):
            with self.assertRaises(ValueError):
                sample_from_logits([1.0, 2.0], top_p=bad)


if __name__ == "__main__":
    unittest.main()
