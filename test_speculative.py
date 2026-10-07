import random
import unittest
from collections import Counter

from language import CharTransformerModel
from speculative import _distributions, speculative_generate


def model(seed: int, width: int = 8) -> CharTransformerModel:
    return CharTransformerModel(5, context_size=12, embedding_dim=width, num_heads=2, num_layers=1, seed=seed)


class SpeculativeTests(unittest.TestCase):
    def test_greedy_output_equals_the_target_alone(self) -> None:
        target, draft = model(1, 16), model(2)
        prompt = [0, 3, 1]
        tokens, stats = speculative_generate(target, draft, prompt, length=8, context_size=12,
                                             lookahead=3, temperature=0.0, rng=random.Random(0))
        expected = list(prompt)
        for _ in range(8):
            p = _distributions(target, expected, 1, 12, 0.0)[0]
            expected.append(p.index(1.0))
        self.assertEqual(tokens, expected)
        self.assertLessEqual(stats["target_passes"], 8)

    def test_identical_draft_is_always_accepted(self) -> None:
        target = model(3)
        _, stats = speculative_generate(target, target, [1], length=9, context_size=12,
                                        lookahead=4, temperature=1.0, rng=random.Random(1))
        self.assertEqual(stats["accepted"], stats["drafted"])
        self.assertEqual(stats["target_passes"], 2)          # 4 + 1, then 4 more (trimmed to 9)

    def test_tokens_follow_the_target_distribution(self) -> None:
        target, draft = model(4), model(5)
        prompt = [2, 4]
        p = _distributions(target, prompt, 1, 12, 1.0)[0]
        rng = random.Random(6)
        trials = 3000
        counts = Counter(
            speculative_generate(target, draft, prompt, length=1, context_size=12, lookahead=1, rng=rng)[0][-1]
            for _ in range(trials)
        )
        for token in range(5):
            self.assertAlmostEqual(counts[token] / trials, p[token], delta=0.03)


if __name__ == "__main__":
    unittest.main()
