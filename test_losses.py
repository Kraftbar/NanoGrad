import unittest

from losses import binary_cross_entropy, mse_loss, soft_cross_entropy, softmax_cross_entropy
from tensor import Tensor


class TensorLossTests(unittest.TestCase):
    def test_mse_loss(self) -> None:
        predictions = Tensor.from_list([1, 2, 3])
        targets = Tensor.from_list([1, 4, 5])

        loss = mse_loss(predictions, targets)

        self.assertEqual(loss.shape, (1,))
        self.assertAlmostEqual(loss[0], 8 / 3)

    def test_mse_loss_2d(self) -> None:
        predictions = Tensor.from_list([
            [1, 2],
            [3, 4],
        ])
        targets = Tensor.from_list([
            [1, 1],
            [5, 4],
        ])

        loss = mse_loss(predictions, targets)

        self.assertEqual(loss.shape, (1,))
        self.assertAlmostEqual(loss[0], 1.25)

    def test_binary_cross_entropy_prefers_good_probabilities(self) -> None:
        good = binary_cross_entropy(
            probabilities=Tensor.from_list([0.01, 0.99]),
            targets=Tensor.from_list([0, 1]),
        )
        bad = binary_cross_entropy(
            probabilities=Tensor.from_list([0.99, 0.01]),
            targets=Tensor.from_list([0, 1]),
        )

        self.assertLess(good[0], bad[0])

    def test_binary_cross_entropy_2d(self) -> None:
        loss = binary_cross_entropy(
            probabilities=Tensor.from_list([
                [0.01, 0.99],
                [0.90, 0.10],
            ]),
            targets=Tensor.from_list([
                [0, 1],
                [1, 0],
            ]),
        )

        self.assertEqual(loss.shape, (1,))
        self.assertLess(loss[0], 0.1)

    def test_softmax_cross_entropy_prefers_correct_logits(self) -> None:
        good = softmax_cross_entropy(
            logits=Tensor.from_list([
                [5.0, 1.0, -1.0],
                [0.5, 4.0, 0.0],
            ]),
            targets=[0, 1],
        )
        bad = softmax_cross_entropy(
            logits=Tensor.from_list([
                [-1.0, 1.0, 5.0],
                [4.0, 0.5, 0.0],
            ]),
            targets=[0, 1],
        )

        self.assertLess(good[0], bad[0])

    def test_softmax_cross_entropy_gradient_matches_finite_difference(self) -> None:
        logits = Tensor.from_list([
            [0.5, -1.0, 2.0],
            [1.5, 0.0, -0.5],
        ], requires_grad=True)
        targets = Tensor.from_list([2, 0])
        loss = softmax_cross_entropy(logits, targets)

        loss.backward()

        self.assertIsNotNone(logits.grad)
        for i, grad in enumerate(logits.grad or []):
            numerical = finite_difference(
                logits,
                i,
                lambda: softmax_cross_entropy(logits, targets)[0],
            )
            self.assertAlmostEqual(grad, numerical, places=5)

    def test_loss_shape_errors(self) -> None:
        with self.assertRaises(ValueError):
            mse_loss(
                Tensor.from_list([1, 2]),
                Tensor.from_list([1, 2, 3]),
            )

        with self.assertRaises(ValueError):
            binary_cross_entropy(
                Tensor.from_list([0.5, 0.5]),
                Tensor.from_list([0, 1, 1]),
            )

        with self.assertRaises(ValueError):
            softmax_cross_entropy(
                Tensor.from_list([1, 2, 3]),
                [0],
            )

    def test_binary_cross_entropy_domain_errors(self) -> None:
        with self.assertRaises(ValueError):
            binary_cross_entropy(
                Tensor.from_list([-0.1, 0.5]),
                Tensor.from_list([0, 1]),
            )

        with self.assertRaises(ValueError):
            binary_cross_entropy(
                Tensor.from_list([0.5, 0.5]),
                Tensor.from_list([0, 0.25]),
            )

    def test_softmax_cross_entropy_target_errors(self) -> None:
        logits = Tensor.from_list([
            [1, 2, 3],
            [1, 2, 3],
        ])

        with self.assertRaises(ValueError):
            softmax_cross_entropy(logits, [0])

        with self.assertRaises(ValueError):
            softmax_cross_entropy(logits, [0, 3])

        with self.assertRaises(ValueError):
            softmax_cross_entropy(logits, Tensor.from_list([[0, 1]]))


def finite_difference(tensor: Tensor, index: int, loss_fn, eps: float = 1e-6) -> float:
    original = tensor.data[index]

    tensor.data[index] = original + eps
    plus = loss_fn()

    tensor.data[index] = original - eps
    minus = loss_fn()

    tensor.data[index] = original
    return (plus - minus) / (2 * eps)


class StableCrossEntropyTests(unittest.TestCase):
    def test_confident_wrong_prediction_gives_a_finite_loss(self) -> None:
        import math

        from losses import softmax_cross_entropy
        from tensor import Tensor

        logits = Tensor([0.0, 900.0], (1, 2), requires_grad=True)     # p(target) underflows to 0
        loss = softmax_cross_entropy(logits, [0])
        self.assertAlmostEqual(loss.item(), 900.0 + math.log1p(math.exp(-900.0)), places=9)
        loss.backward()
        self.assertEqual(list(logits.grad), [-1.0, 1.0])


class WeightedCrossEntropyTests(unittest.TestCase):
    def test_matches_a_weighted_average_and_its_gradient(self) -> None:
        import random

        from test_gradcheck import max_gradient_error

        rng = random.Random(3)
        x = Tensor([rng.uniform(-2, 2) for _ in range(15)], (5, 3), requires_grad=True)
        targets, weights = [0, 2, 2, 1, 0], [0.5, 2.0, 1.0]
        per_row = [softmax_cross_entropy(Tensor(list(x.data[3 * r : 3 * r + 3]), (1, 3)), [t]).item()
                   for r, t in enumerate(targets)]
        expected = sum(weights[t] * l for t, l in zip(targets, per_row)) / sum(weights[t] for t in targets)
        got = softmax_cross_entropy(x, targets, class_weights=weights)
        self.assertAlmostEqual(got.item(), expected, places=12)
        self.assertLess(max_gradient_error(lambda: softmax_cross_entropy(x, targets, class_weights=weights), [x]),
                        1e-8)
        same = softmax_cross_entropy(x, targets, class_weights=[1.0, 1.0, 1.0]).item()
        self.assertAlmostEqual(same, softmax_cross_entropy(x, targets).item(), places=12)


class SoftCrossEntropyTests(unittest.TestCase):
    def test_one_hot_targets_give_the_usual_cross_entropy(self) -> None:
        values = [0.3, -1.2, 2.0, 0.5, 0.1, -0.4]
        hard = softmax_cross_entropy(Tensor(values, (2, 3)), [2, 0]).item()
        soft = soft_cross_entropy(Tensor(values, (2, 3)), Tensor([0, 0, 1, 1, 0, 0], (2, 3))).item()
        self.assertAlmostEqual(hard, soft, places=12)

    def test_gradient_is_softmax_minus_targets(self) -> None:
        import random

        from test_gradcheck import max_gradient_error

        rng = random.Random(0)
        x = Tensor([rng.uniform(-3, 3) for _ in range(12)], (3, 4), requires_grad=True)
        q = Tensor([0.1, 0.2, 0.3, 0.4, 0, 0, 1, 0, 0.25, 0.25, 0.25, 0.25], (3, 4))
        self.assertLess(max_gradient_error(lambda: soft_cross_entropy(x, q), [x]), 1e-8)


if __name__ == "__main__":
    unittest.main()


class LinearCrossEntropyTests(unittest.TestCase):
    """The fused output layer + loss against linear() then softmax_cross_entropy()."""

    def setUp(self) -> None:
        import random

        rng = random.Random(5)
        self.rows, self.features, self.vocab = 600, 12, 301           # 600 rows: chunks of 256, 256, 88
        self.x = [rng.uniform(-1, 1) for _ in range(self.rows * self.features)]
        self.w = [rng.uniform(-0.5, 0.5) for _ in range(self.vocab * self.features)]
        self.b = [rng.uniform(-0.5, 0.5) for _ in range(self.vocab)]
        self.targets = [rng.randrange(self.vocab) for _ in range(self.rows)]

    def run_both(self, scale=1.0, tied=False, chunk_rows=256):
        from losses import linear_cross_entropy
        from tensor_nn import linear

        results = []
        for fused in (False, True):
            x = Tensor(self.x, (20, 30, self.features), requires_grad=True)
            w = Tensor(self.w, (self.vocab, self.features), requires_grad=True)
            b = Tensor(self.b, (self.vocab,), requires_grad=True)
            if fused:
                loss = linear_cross_entropy(x, w, b, self.targets, chunk_rows=chunk_rows)
            else:
                logits = linear(x, w, b).reshape((self.rows, self.vocab))
                loss = softmax_cross_entropy(logits, self.targets)
            if tied:                                                  # the weight is also used elsewhere
                loss = loss + (w * w).sum()
            (loss * scale).backward()
            results.append([loss.item(), list(x.grad), list(w.grad), list(b.grad)])
        return results

    def test_same_bits_as_linear_then_softmax_cross_entropy(self) -> None:
        import kernels

        unfused, fused = self.run_both()
        if kernels.BACKEND == "c":                                   # same operations, same order
            self.assertEqual(fused, unfused)
        else:                                                        # the reference product sums differently
            for left, right in zip(fused[1:], unfused[1:]):
                for a, b in zip(left, right):
                    self.assertAlmostEqual(a, b, places=12)
            self.assertAlmostEqual(fused[0], unfused[0], places=12)

    def test_default_chunks_and_one_row_chunks_agree(self) -> None:
        reference = self.run_both(chunk_rows=None)[0]
        for chunk_rows in (None, 1, 600, 10_000):
            fused = self.run_both(chunk_rows=chunk_rows)[1]
            self.assertAlmostEqual(fused[0], reference[0], places=12)
            for left, right in zip(fused[1:], reference[1:]):
                for a, b in zip(left, right):
                    self.assertAlmostEqual(a, b, places=12)

    def test_other_upstream_gradients_and_shared_weights(self) -> None:
        unfused, fused = self.run_both(scale=0.37, tied=True)
        self.assertAlmostEqual(fused[0], unfused[0], places=12)
        for left, right in zip(fused[1:], unfused[1:]):
            for a, b in zip(left, right):
                self.assertAlmostEqual(a, b, places=12)

    def test_without_gradients_only_the_loss(self) -> None:
        from losses import linear_cross_entropy
        from tensor import no_grad

        x = Tensor(self.x, (self.rows, self.features), requires_grad=True)
        w = Tensor(self.w, (self.vocab, self.features), requires_grad=True)
        b = Tensor(self.b, (self.vocab,), requires_grad=True)
        with no_grad():
            loss = linear_cross_entropy(x, w, b, self.targets)
        self.assertFalse(loss.requires_grad)
        self.assertAlmostEqual(loss.item(), self.run_both()[0][0], places=12)
