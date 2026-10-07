import random
import unittest

from bert import TinyBERT
from lora import LoRALinear, add_lora, merge_lora
from optim import TensorAdam
from tensor import Tensor
from tensor_nn import TensorLinear


def tokens(seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(9) for _ in range(6)] for _ in range(2)]


class LoRATests(unittest.TestCase):
    def test_starts_as_the_frozen_layer_and_trains_only_the_adapter(self) -> None:
        base = TensorLinear(5, 3, seed=1)
        before = list(base.weight.data)
        layer = LoRALinear(base, rank=2)
        x = Tensor([random.Random(2).uniform(-1, 1) for _ in range(4 * 5)], (4, 5))
        self.assertEqual(list(layer(x).data), list(base(x).data))       # B = 0 at the start
        optimizer = TensorAdam(layer.parameters(), lr=0.05)
        for _ in range(5):
            optimizer.zero_grad()
            (layer(x) * layer(x)).sum().backward()
            optimizer.step()
        self.assertEqual(list(base.weight.data), before)
        self.assertNotEqual(list(layer(x).data), list(base(x).data))
        self.assertEqual(sum(p.numel for p in layer.parameters()), 2 * 5 + 3 * 2)

    def test_add_lora_freezes_the_model_and_merging_keeps_its_outputs(self) -> None:
        model = TinyBERT(9, context_size=6, embedding_dim=8, num_heads=2, num_layers=2, seed=3)
        original = list(model.encode(tokens(4)).data)
        trainable = add_lora(model, rank=2)
        self.assertEqual(len(trainable), 2 * 2 * 2 * 2)                 # layers x (q, v) x heads x (A, B)
        self.assertEqual(list(model.encode(tokens(4)).data), original)
        optimizer = TensorAdam(trainable, lr=0.05)
        for _ in range(3):
            optimizer.zero_grad()
            model.encode(tokens(5)).sum().backward()
            optimizer.step()
        adapted = list(model.encode(tokens(4)).data)
        self.assertNotEqual(adapted, original)
        merge_lora(model)
        for a, b in zip(model.encode(tokens(4)).data, adapted):
            self.assertAlmostEqual(a, b, places=12)

    def test_rank_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            LoRALinear(TensorLinear(2, 2), rank=0)


if __name__ == "__main__":
    unittest.main()
