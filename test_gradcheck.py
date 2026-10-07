import math
import random
import unittest

from gradcheck import max_gradient_error
from language import CausalSelfAttention, CharTransformerModel
from losses import softmax_cross_entropy
from tensor import (
    Tensor,
    avg_pool2d,
    concat,
    conv2d_valid,
    gather_rows,
    matmul,
    max_pool2d,
    no_grad,
    permute,
    stack,
)
from tensor_nn import TensorLayerNorm, layer_norm


TOLERANCE = 1e-5


def random_tensor(shape, *, seed, requires_grad=True):
    rng = random.Random(seed)
    count = math.prod(shape)
    return Tensor(
        [rng.uniform(-1.0, 1.0) for _ in range(count)],
        shape,
        requires_grad=requires_grad,
    )


def weighted_loss(output: Tensor, *, seed: int = 99) -> Tensor:
    """Scalar loss whose gradient differs per output value."""

    weights = random_tensor(output.shape, seed=seed, requires_grad=False)
    return (output * weights).sum()


class GradientCheckTests(unittest.TestCase):
    def assertGradientsMatch(self, build, tensors) -> None:
        error = max_gradient_error(lambda: weighted_loss(build()), tensors)
        self.assertLess(error, TOLERANCE)

    def test_broadcast_elementwise_ops(self) -> None:
        left = random_tensor((2, 3, 4), seed=1)
        row = random_tensor((4,), seed=2)
        column = random_tensor((3, 1), seed=3)

        self.assertGradientsMatch(lambda: left + row, [left, row])
        self.assertGradientsMatch(lambda: left - column, [left, column])
        self.assertGradientsMatch(lambda: left * row * column, [left, row, column])
        self.assertGradientsMatch(lambda: row - left, [left, row])

    def test_unary_ops(self) -> None:
        values = random_tensor((3, 4), seed=4)
        positive = Tensor(
            [abs(value) + 0.5 for value in values.data],
            values.shape,
            requires_grad=True,
        )

        for op in ("exp", "relu", "gelu", "sigmoid", "tanh"):
            with self.subTest(op=op):
                self.assertGradientsMatch(
                    lambda op=op: getattr(values, op)(),
                    [values],
                )
        self.assertGradientsMatch(lambda: positive.log(), [positive])
        self.assertGradientsMatch(lambda: positive.reciprocal(), [positive])
        self.assertGradientsMatch(lambda: 1.5 / positive, [positive])

    def test_matmul_shapes(self) -> None:
        cases = [
            ((4,), (4,)),
            ((3, 4), (4,)),
            ((4,), (4, 2)),
            ((3, 4), (4, 2)),
            ((2, 3, 4), (4, 2)),
            ((2, 3, 4), (2, 4, 5)),
            ((3, 4), (2, 4, 5)),
        ]
        for seed, (left_shape, right_shape) in enumerate(cases):
            with self.subTest(left=left_shape, right=right_shape):
                left = random_tensor(left_shape, seed=seed)
                right = random_tensor(right_shape, seed=seed + 50)
                self.assertGradientsMatch(lambda: matmul(left, right), [left, right])

    def test_batched_matmul_matches_per_batch_products(self) -> None:
        left = random_tensor((2, 3, 4), seed=5, requires_grad=False)
        right = random_tensor((2, 4, 2), seed=6, requires_grad=False)

        batched = matmul(left, right)

        for batch in range(2):
            single = matmul(
                Tensor(left.data[batch * 12 : (batch + 1) * 12], (3, 4)),
                Tensor(right.data[batch * 8 : (batch + 1) * 8], (4, 2)),
            )
            for expected, actual in zip(single.data, batched.data[batch * 6 :]):
                self.assertAlmostEqual(expected, actual)

    def test_matmul_rejects_mismatched_batches(self) -> None:
        with self.assertRaises(ValueError):
            matmul(Tensor.zeros((2, 3, 4)), Tensor.zeros((3, 4, 2)))

    def test_shape_ops(self) -> None:
        values = random_tensor((2, 3, 4), seed=7)
        other = random_tensor((2, 2, 4), seed=8)
        same = random_tensor((2, 3, 4), seed=9)

        self.assertGradientsMatch(lambda: permute(values, (2, 0, 1)), [values])
        self.assertGradientsMatch(lambda: values.reshape((6, 4)), [values])
        self.assertGradientsMatch(
            lambda: concat([values, other], axis=1),
            [values, other],
        )
        self.assertGradientsMatch(
            lambda: stack([values, same], axis=2),
            [values, same],
        )

    def test_concat_values(self) -> None:
        left = Tensor.from_list([[1, 2], [3, 4]])
        right = Tensor.from_list([[5], [6]])

        self.assertEqual(concat([left, right], axis=1).tolist(), [[1, 2, 5], [3, 4, 6]])
        self.assertEqual(concat([left, left], axis=0).shape, (4, 2))
        with self.assertRaises(ValueError):
            concat([left, right], axis=0)

    def test_gather_rows_accumulates_repeated_ids(self) -> None:
        table = random_tensor((5, 3), seed=10)

        self.assertGradientsMatch(lambda: gather_rows(table, [4, 0, 4, 2]), [table])

        gather_rows(table, [1, 1]).sum().backward()
        self.assertEqual(list(table.grad[3:6]), [2.0, 2.0, 2.0])
        with self.assertRaises(ValueError):
            gather_rows(table, [5])

    def test_reductions(self) -> None:
        values = random_tensor((2, 3, 4), seed=11)

        for axis in (0, 1, 2, -1):
            with self.subTest(axis=axis):
                self.assertGradientsMatch(lambda axis=axis: values.sum(axis=axis), [values])
                self.assertGradientsMatch(lambda axis=axis: values.mean(axis=axis), [values])
                self.assertGradientsMatch(
                    lambda axis=axis: values.softmax(axis=axis),
                    [values],
                )

    def test_convolution_shapes(self) -> None:
        cases = [
            ((5, 5), (2, 3)),
            ((5, 5), (2, 3, 3)),
            ((2, 5, 4), (2, 2, 2)),
            ((2, 5, 4), (3, 2, 2, 2)),
            ((2, 2, 5, 4), (3, 2, 3, 2)),
        ]
        for seed, (image_shape, kernel_shape) in enumerate(cases):
            with self.subTest(image=image_shape, kernel=kernel_shape):
                image = random_tensor(image_shape, seed=seed + 20)
                kernel = random_tensor(kernel_shape, seed=seed + 30)
                self.assertGradientsMatch(
                    lambda: conv2d_valid(image, kernel),
                    [image, kernel],
                )

    def test_single_filter_convolution_drops_filter_axis(self) -> None:
        self.assertEqual(conv2d_valid(Tensor.zeros((4, 5)), Tensor.zeros((1, 2, 2))).shape, (3, 4))
        self.assertEqual(conv2d_valid(Tensor.zeros((4, 5)), Tensor.zeros((2, 2, 2))).shape, (2, 3, 4))
        self.assertEqual(
            conv2d_valid(Tensor.zeros((2, 4, 5)), Tensor.zeros((1, 2, 2, 2))).shape,
            (3, 4),
        )
        self.assertEqual(
            conv2d_valid(Tensor.zeros((1, 2, 4, 5)), Tensor.zeros((1, 2, 2, 2))).shape,
            (1, 1, 3, 4),
        )

    def test_pooling(self) -> None:
        image = random_tensor((2, 2, 5, 6), seed=12)

        for stride in (None, (1, 2)):
            with self.subTest(stride=stride):
                self.assertGradientsMatch(
                    lambda stride=stride: avg_pool2d(image, (2, 3), stride=stride),
                    [image],
                )
                self.assertGradientsMatch(
                    lambda stride=stride: max_pool2d(image, (2, 2), stride=stride),
                    [image],
                )

    def test_fused_layer_norm_matches_composed_ops(self) -> None:
        inputs = random_tensor((2, 3, 4), seed=13)
        weight = random_tensor((4,), seed=14)
        bias = random_tensor((4,), seed=15)

        def composed() -> Tensor:
            mean = inputs.mean(axis=-1).reshape((2, 3, 1))
            centered = inputs - mean
            variance = (centered * centered).mean(axis=-1).reshape((2, 3, 1))
            std = ((variance + 1e-5).log() * 0.5).exp()
            return (centered / std) * weight + bias

        fused = layer_norm(inputs, weight, bias, eps=1e-5)
        for expected, actual in zip(composed().data, fused.data):
            self.assertAlmostEqual(expected, actual)

        weighted_loss(composed()).backward()
        expected_grads = [inputs.grad[:], weight.grad[:], bias.grad[:]]
        weighted_loss(layer_norm(inputs, weight, bias, eps=1e-5)).backward()
        for expected, tensor in zip(expected_grads, [inputs, weight, bias]):
            for expected_value, actual_value in zip(expected, tensor.grad):
                self.assertAlmostEqual(expected_value, actual_value)

        self.assertGradientsMatch(
            lambda: layer_norm(inputs, weight, bias),
            [inputs, weight, bias],
        )
        layer = TensorLayerNorm(4)
        self.assertGradientsMatch(lambda: layer(inputs), [inputs, *layer.parameters()])

    def test_attention_and_transformer_gradients(self) -> None:
        attention = CausalSelfAttention(4, 3, num_heads=2, seed=1)
        inputs = random_tensor((2, 3, 4), seed=16)

        self.assertGradientsMatch(
            lambda: attention(inputs),
            [inputs, *attention.parameters()],
        )

        model = CharTransformerModel(
            vocab_size=5,
            context_size=3,
            embedding_dim=4,
            hidden_dim=6,
            num_heads=2,
            num_layers=2,
            feed_forward_activation="gelu",
            tie_embeddings=True,
            seed=2,
        )
        tokens = Tensor.from_list([[0, 1, 2], [4, 3, 1]])
        targets = [1, 2, 3, 3, 1, 0]

        def loss() -> Tensor:
            logits = model.sequence_logits(tokens)
            return softmax_cross_entropy(logits.reshape((6, 5)), targets)

        error = max_gradient_error(loss, model.parameters())
        self.assertLess(error, TOLERANCE)

    def test_no_grad_skips_graph_recording(self) -> None:
        weight = random_tensor((3, 3), seed=17)

        with no_grad():
            out = matmul(weight, weight).relu().sum()

        self.assertFalse(out.requires_grad)
        self.assertEqual(out._prev, set())

        out = matmul(weight, weight).sum()
        self.assertTrue(out.requires_grad)

    def test_deep_graph_backward_does_not_hit_recursion_limit(self) -> None:
        value = Tensor.from_list([1.0], requires_grad=True)
        out = value
        for _ in range(5000):
            out = out * 1.0

        out.backward()

        self.assertEqual(list(value.grad), [1.0])


    def test_backward_frees_the_graph_unless_retained(self) -> None:
        weight = random_tensor((3,), seed=18)
        out = (weight * weight).sum()

        out.backward(retain_graph=True)
        first = list(weight.grad)
        out.backward()
        self.assertEqual(list(weight.grad), first)

        with self.assertRaises(RuntimeError):
            out.backward()
        # A new graph from the same leaves works as usual.
        (weight * 3.0).sum().backward()
        self.assertEqual(list(weight.grad), [3.0, 3.0, 3.0])


if __name__ == "__main__":
    unittest.main()
