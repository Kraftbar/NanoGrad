import random
import unittest

from language import CharTransformerModel
from quantize import dequantize, quantize, quantize_model, quantized_matrices


class QuantizeTests(unittest.TestCase):
    def test_round_trip_error_is_at_most_half_a_step(self) -> None:
        rng = random.Random(0)
        values = [rng.gauss(0, 1) for _ in range(100)]
        for bits in (8, 4):
            codes, scales = quantize(values, bits=bits, group_size=16)
            self.assertEqual(len(scales), 7)
            self.assertTrue(all(abs(code) <= 2 ** (bits - 1) - 1 for code in codes))
            restored = dequantize(codes, scales, group_size=16)
            for index, (value, back) in enumerate(zip(values, restored)):
                self.assertLessEqual(abs(value - back), scales[index // 16] / 2 + 1e-12)

    def test_zero_group_and_extremes_are_exact(self) -> None:
        codes, scales = quantize([0.0, 0.0, -2.0, 1.0], bits=4, group_size=2)
        self.assertEqual(dequantize(codes, scales, group_size=2)[:3], [0.0, 0.0, -2.0])

    def test_quantize_model_touches_only_matrices_and_reports_sizes(self) -> None:
        model = CharTransformerModel(11, context_size=4, embedding_dim=8, num_heads=2, num_layers=1, seed=3)
        vectors = {id(p): list(p.data) for p in model.parameters() if p.ndim == 1}
        sizes = quantize_model(model, bits=8, group_size=4)
        for parameter in model.parameters():
            if parameter.ndim == 1:
                self.assertEqual(list(parameter.data), vectors[id(parameter)])
        count = sum(weight.numel for weight in quantized_matrices(model))
        self.assertEqual(sizes["weights"], count)
        self.assertAlmostEqual(sizes["quantized_bytes"], count + 4 * count / 4)


if __name__ == "__main__":
    unittest.main()
