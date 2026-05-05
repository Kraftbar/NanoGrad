import unittest

from losses import binary_cross_entropy, mse_loss, softmax_cross_entropy
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


if __name__ == "__main__":
    unittest.main()
