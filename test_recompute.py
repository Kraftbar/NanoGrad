import random
import unittest

from language import CharTransformerModel, TransformerBlock
from recompute import recompute
from tensor import Tensor


def inputs(seed: int) -> Tensor:
    rng = random.Random(seed)
    return Tensor([rng.uniform(-1, 1) for _ in range(2 * 5 * 8)], (2, 5, 8), requires_grad=True)


class RecomputeTests(unittest.TestCase):
    def test_same_outputs_and_gradients_as_running_the_block(self) -> None:
        for dropout in (0.0, 0.3):
            with self.subTest(dropout=dropout):
                results = []
                for use_recompute in (False, True):
                    block = TransformerBlock(8, 5, num_heads=2, dropout=dropout, seed=3)
                    x = inputs(4)
                    out = recompute(block, x) if use_recompute else block(x)
                    first = recompute(block, out) if use_recompute else block(out)   # two blocks deep
                    (first * first).sum().backward()
                    results.append((list(first.data), list(x.grad), [list(p.grad) for p in block.parameters()]))
                (out_a, x_a, p_a), (out_b, x_b, p_b) = results
                self.assertEqual(out_a, out_b)
                for a, b in zip(x_a, x_b):
                    self.assertAlmostEqual(a, b, places=12)
                for left, right in zip(p_a, p_b):
                    for a, b in zip(left, right):
                        self.assertAlmostEqual(a, b, places=12)

    def test_gpt_option_trains_the_same(self) -> None:
        settings = dict(context_size=6, embedding_dim=8, num_heads=2, num_layers=2, seed=5)
        grads = []
        for flag in (False, True):
            model = CharTransformerModel(9, recompute_blocks=flag, **settings)
            logits = model.sequence_logits(Tensor.from_list([[1, 2, 3, 4]]))
            (logits * logits).sum().backward()
            grads.append([list(p.grad) for p in model.parameters()])
        for left, right in zip(*grads):
            for a, b in zip(left, right):
                self.assertAlmostEqual(a, b, places=12)


if __name__ == "__main__":
    unittest.main()
