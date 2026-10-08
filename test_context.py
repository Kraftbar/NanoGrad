import copy
import unittest

from language import CharTransformerModel
from tensor import Tensor, no_grad


class ContextExtensionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = CharTransformerModel(9, context_size=8, embedding_dim=16, num_heads=2, positions="rope",
                                          norm="rms", seed=4)

    def test_extrapolating_keeps_the_first_positions(self) -> None:
        tokens = [[1, 4, 2, 8, 5, 7, 3, 0]]
        longer = copy.deepcopy(self.model)
        longer.extend_context(16, interpolate=False)
        with no_grad():
            short = self.model.sequence_logits(Tensor.from_list(tokens)).data
            extended = longer.sequence_logits(Tensor.from_list([tokens[0] + [6] * 8])).data
        # Causal attention: the first 8 positions see nothing new and use the same angles.
        for a, b in zip(short, extended[: len(short)]):
            self.assertAlmostEqual(a, b, places=12)

    def test_interpolation_reuses_the_trained_angles(self) -> None:
        # Position interpolation by 2 on tokens repeated twice: positions 0, 0.5, 1, 1.5, ...
        # so the even positions see the trained angles at 0, 1, 2, ... (only the odd ones are new).
        longer = copy.deepcopy(self.model)
        longer.extend_context(16, interpolate=True)
        for block in longer.blocks:
            self.assertEqual(block.attention.rope_scale, 0.5)
        with no_grad():
            out = longer.sequence_logits(Tensor.from_list([list(range(9)) + list(range(7))]))
        self.assertEqual(out.shape, (1, 16, 9))

    def test_learned_positions_refuse(self) -> None:
        model = CharTransformerModel(9, context_size=8, embedding_dim=16, seed=1)
        with self.assertRaises(ValueError):
            model.extend_context(16)

    def test_kv_cache_matches_full_pass_after_interpolation(self) -> None:
        longer = copy.deepcopy(self.model)
        longer.extend_context(16, interpolate=True)
        tokens = [3, 1, 4, 1, 5, 2, 6, 5, 3, 5, 8, 0]
        with no_grad():
            full = longer.sequence_logits(Tensor.from_list([tokens])).data
            cache = longer.new_cache()
            for position, token in enumerate(tokens):
                step = longer.step_logits(token, position, cache).data
                for a, b in zip(step, full[position * 9 : (position + 1) * 9]):
                    self.assertAlmostEqual(a, b, places=10)


if __name__ == "__main__":
    unittest.main()
