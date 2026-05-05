import unittest

from datasets import and_gate, or_gate, xor_gate
from tensor import Tensor, matmul
from tensor_nn import binary_mlp, linear
from train import train_tensor_binary_classifier


class TensorAutogradTests(unittest.TestCase):
    def test_elementwise_gradients(self) -> None:
        x = Tensor.from_list([1.0, 2.0, 3.0], requires_grad=True)
        y = ((x * x) + (x * 2)).mean()

        y.backward()

        self.assertEqual(y.shape, (1,))
        self.assertIsNotNone(x.grad)
        self.assertEqual(
            x.grad,
            [
                4.0 / 3.0,
                6.0 / 3.0,
                8.0 / 3.0,
            ],
        )

    def test_matmul_gradients_match_finite_difference(self) -> None:
        weight = Tensor.from_list([
            [1.0, -2.0],
            [0.5, 3.0],
        ], requires_grad=True)
        inputs = Tensor.from_list([
            [2.0, -1.0],
            [0.0, 4.0],
        ], requires_grad=True)

        loss = matmul(inputs, weight).sum()
        loss.backward()

        def loss_fn() -> float:
            return matmul(inputs, weight).sum()[0]

        self.assert_grad_close(weight, loss_fn)
        self.assert_grad_close(inputs, loss_fn)

    def test_row_broadcast_bias_gradient(self) -> None:
        inputs = Tensor.from_list([
            [1.0, 2.0],
            [3.0, 4.0],
        ])
        bias = Tensor.from_list([10.0, 20.0], requires_grad=True)

        loss = (inputs + bias).sum()
        loss.backward()

        self.assertEqual(bias.grad, [2.0, 2.0])

    def test_log_sigmoid_gradient_matches_finite_difference(self) -> None:
        x = Tensor.from_list([0.25, -0.5], requires_grad=True)
        loss = (x.sigmoid().log() * Tensor.from_list([2.0, -1.0])).sum()

        loss.backward()

        def loss_fn() -> float:
            return (x.sigmoid().log() * Tensor.from_list([2.0, -1.0])).sum()[0]

        self.assert_grad_close(x, loss_fn)

    def test_tensor_logistic_regression_learns_linear_logic_gates(self) -> None:
        for dataset in (and_gate, or_gate):
            with self.subTest(dataset=dataset.__name__):
                xs, ys = dataset()
                inputs = Tensor.from_list(xs)
                targets = Tensor.from_list([[y] for y in ys])
                weight = Tensor.from_list([[0.0, 0.0]], requires_grad=True)
                bias = Tensor.from_list([0.0], requires_grad=True)

                summary = train_tensor_binary_classifier(
                    lambda: linear(inputs, weight, bias).sigmoid(),
                    targets,
                    [weight, bias],
                    steps=800,
                    lr=0.5,
                )

                self.assertLess(summary.final_loss, summary.initial_loss)
                self.assertLess(summary.final_loss, 0.1)
                self.assertEqual(summary.accuracy, 1.0)

    def test_tensor_mlp_learns_xor(self) -> None:
        xs, ys = xor_gate()
        inputs = Tensor.from_list(xs)
        targets = Tensor.from_list([[y] for y in ys])
        hidden_weight = Tensor.from_list([
            [1.0, -1.0],
            [-1.0, 1.0],
        ], requires_grad=True)
        hidden_bias = Tensor.from_list([0.0, 0.0], requires_grad=True)
        output_weight = Tensor.from_list([[0.1, 0.1]], requires_grad=True)
        output_bias = Tensor.from_list([0.0], requires_grad=True)

        summary = train_tensor_binary_classifier(
            lambda: binary_mlp(
                inputs,
                hidden_weight,
                hidden_bias,
                output_weight,
                output_bias,
            ),
            targets,
            [
                hidden_weight,
                hidden_bias,
                output_weight,
                output_bias,
            ],
            steps=2000,
            lr=0.5,
        )

        self.assertLess(summary.final_loss, summary.initial_loss)
        self.assertLess(summary.final_loss, 0.01)
        self.assertEqual(summary.accuracy, 1.0)

    def assert_grad_close(self, tensor: Tensor, loss_fn) -> None:
        self.assertIsNotNone(tensor.grad)

        for i, grad in enumerate(tensor.grad or []):
            numerical = finite_difference(tensor, i, loss_fn)
            self.assertAlmostEqual(grad, numerical, places=5)


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
