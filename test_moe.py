import math
import random
import unittest

from moe import MoEFeedForward
from tensor import Tensor, scatter_add_rows
from test_gradcheck import max_gradient_error


def inputs(seed: int, shape=(2, 3, 6)) -> Tensor:
    rng = random.Random(seed)
    return Tensor([rng.uniform(-1, 1) for _ in range(math.prod(shape))], shape, requires_grad=True)


class MoETests(unittest.TestCase):
    def test_each_token_mixes_its_top_k_experts(self) -> None:
        layer = MoEFeedForward(6, 8, num_experts=4, top_k=2, seed=1)
        x = inputs(2)
        out = layer(x)
        self.assertEqual(out.shape, x.shape)
        for n in range(6):
            row = Tensor(list(x.data[n * 6 : (n + 1) * 6]), (1, 6))
            p = list(layer.router(row).softmax(axis=-1).data)
            top = sorted(range(4), key=lambda e: -p[e])[:2]
            expected = [0.0] * 6
            for e in top:
                y = layer.experts[e](row).data
                expected = [total + p[e] / (p[top[0]] + p[top[1]]) * value for total, value in zip(expected, y)]
            for a, b in zip(out.data[n * 6 : (n + 1) * 6], expected):
                self.assertAlmostEqual(a, b, places=12)

    def test_gradients_match_finite_differences(self) -> None:
        for activation in ("gelu", "swiglu"):
            layer = MoEFeedForward(6, 5, num_experts=3, top_k=2, activation=activation, seed=4)
            x = inputs(5, (4, 6))
            weights = Tensor([random.Random(6).uniform(-1, 1) for _ in range(24)], (4, 6))
            parameters = [x, layer.router.weight, layer.experts[0].feed_forward_in.weight]

            def loss():
                return (layer(x) * weights).sum() + layer.aux_loss

            self.assertLess(max_gradient_error(loss, parameters), 1e-6)

    def test_balance_loss_is_one_when_routing_is_uniform(self) -> None:
        layer = MoEFeedForward(4, 4, num_experts=2, top_k=2)
        layer(inputs(7, (3, 4)))
        self.assertAlmostEqual(layer.aux_loss.item(), 1.0, places=12)    # every token uses both experts

    def test_scatter_add_rows_adds_repeats(self) -> None:
        values = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), requires_grad=True)
        out = scatter_add_rows(values, [2, 0, 2], 4)
        self.assertEqual(list(out.data), [3.0, 4.0, 0.0, 0.0, 6.0, 8.0, 0.0, 0.0])
        (out * Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], (4, 2))).sum().backward()
        self.assertEqual(list(values.grad), [5.0, 6.0, 1.0, 2.0, 5.0, 6.0])


class MoETransformerTests(unittest.TestCase):
    def test_moe_gpt_kv_cache_state_and_training(self) -> None:
        from language import CharTransformerModel
        from parallel import sequence_loss
        from tensor import no_grad

        settings = dict(context_size=6, embedding_dim=8, num_heads=2, num_layers=2, num_experts=3,
                        feed_forward_activation="swiglu", norm="rms", positions="rope")
        model = CharTransformerModel(7, seed=1, **settings)
        tokens = [3, 1, 4, 1, 5, 2]
        with no_grad():
            cache = model.new_cache()
            for position, token in enumerate(tokens):
                step = list(model.step_logits(token, position, cache).data)
                full = list(model(Tensor.from_list([tokens[: position + 1]])).data)
                for a, b in zip(step, full):
                    self.assertAlmostEqual(a, b, places=12)
        copy = CharTransformerModel(7, seed=2, **settings)
        copy.load_state_dict(model.state_dict())
        self.assertEqual(list(copy(Tensor.from_list([tokens])).data), list(model(Tensor.from_list([tokens])).data))
        loss = sequence_loss(model, [tokens[:-1]], [tokens[1:]])
        loss.backward()
        router = model.blocks[0].moe.router.weight
        self.assertTrue(any(value != 0.0 for value in router.grad))
        self.assertIsNotNone(model.auxiliary_loss())


if __name__ == "__main__":
    unittest.main()
