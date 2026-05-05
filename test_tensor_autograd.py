import unittest

from datasets import TinyDataset, and_gate, or_gate, xor_gate
from tensor import Tensor, avg_pool2d, conv2d_valid, matmul
from tensor_nn import TensorLinear, TensorMLP
from train import (
    evaluate_tensor_multiclass_dataset,
    train_tensor_binary_classifier,
    train_tensor_multiclass_classifier,
    train_tensor_multiclass_dataset,
)


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

    def test_nd_elementwise_gradients(self) -> None:
        x = Tensor.from_list([
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            [
                [5.0, 6.0],
                [7.0, 8.0],
            ],
        ], requires_grad=True)
        y = (x * Tensor.from_list([2.0])).sum()

        y.backward()

        self.assertEqual(y.shape, (1,))
        self.assertEqual(x.grad, [2.0] * 8)

    def test_subtraction_and_scalar_reverse_gradients(self) -> None:
        x = Tensor.from_list([1.0, -2.0, 3.0], requires_grad=True)
        y = Tensor.from_list([0.5, 4.0, -1.0], requires_grad=True)
        loss = ((10 - x) + (y - 3) - (x - y)).sum()

        loss.backward()

        self.assertEqual(x.grad, [-2.0, -2.0, -2.0])
        self.assertEqual(y.grad, [2.0, 2.0, 2.0])

    def test_backward_with_explicit_gradient(self) -> None:
        x = Tensor.from_list([1.0, 2.0, 3.0], requires_grad=True)
        y = x * x

        y.backward([1.0, 0.5, -1.0])

        self.assertEqual(x.grad, [2.0, 2.0, -6.0])

    def test_non_scalar_backward_requires_explicit_gradient(self) -> None:
        x = Tensor.from_list([1.0, 2.0], requires_grad=True)

        with self.assertRaises(ValueError):
            x.backward()

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

    def test_vector_matrix_matmul_gradients_match_finite_difference(self) -> None:
        inputs = Tensor.from_list([2.0, -1.0, 0.5], requires_grad=True)
        weight = Tensor.from_list([
            [1.0, -2.0],
            [0.5, 3.0],
            [-1.0, 4.0],
        ], requires_grad=True)

        loss = matmul(inputs, weight).sum()
        loss.backward()

        def loss_fn() -> float:
            return matmul(inputs, weight).sum()[0]

        self.assert_grad_close(inputs, loss_fn)
        self.assert_grad_close(weight, loss_fn)

    def test_reshape_gradient_preserves_flat_order(self) -> None:
        x = Tensor.from_list([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        weights = Tensor.from_list([
            [0.5, -1.0],
            [2.0, 3.0],
        ])
        loss = (x.reshape((2, 2)) * weights).sum()

        loss.backward()

        self.assertEqual(x.grad, [0.5, -1.0, 2.0, 3.0])

    def test_flatten_gradient_preserves_flat_order(self) -> None:
        x = Tensor.from_list([
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            [
                [5.0, 6.0],
                [7.0, 8.0],
            ],
        ], requires_grad=True)
        weights = Tensor.from_list([
            0.5,
            -1.0,
            2.0,
            3.0,
            1.5,
            -0.5,
            0.25,
            4.0,
        ])
        loss = (x.flatten() * weights).sum()

        loss.backward()

        self.assertEqual(x.grad, weights.data)

    def test_conv2d_valid_gradients_match_finite_difference(self) -> None:
        image = Tensor.from_list([
            [1.0, 2.0, -1.0],
            [0.5, -2.0, 3.0],
            [4.0, 1.5, -0.5],
        ], requires_grad=True)
        kernel = Tensor.from_list([
            [0.25, -1.0],
            [2.0, 0.5],
        ], requires_grad=True)

        loss = conv2d_valid(image, kernel).sum()
        loss.backward()

        def loss_fn() -> float:
            return conv2d_valid(image, kernel).sum()[0]

        self.assert_grad_close(image, loss_fn)
        self.assert_grad_close(kernel, loss_fn)

    def test_conv2d_valid_channel_gradients_match_finite_difference(self) -> None:
        image = Tensor.from_list([
            [
                [1.0, 2.0, -1.0],
                [0.5, -2.0, 3.0],
                [4.0, 1.5, -0.5],
            ],
            [
                [0.25, -1.0, 2.0],
                [3.0, 0.5, -2.0],
                [1.0, -0.75, 4.0],
            ],
        ], requires_grad=True)
        kernel = Tensor.from_list([
            [
                [0.25, -1.0],
                [2.0, 0.5],
            ],
            [
                [-0.5, 1.5],
                [0.75, -2.0],
            ],
        ], requires_grad=True)

        loss = conv2d_valid(image, kernel).sum()
        loss.backward()

        def loss_fn() -> float:
            return conv2d_valid(image, kernel).sum()[0]

        self.assert_grad_close(image, loss_fn)
        self.assert_grad_close(kernel, loss_fn)

    def test_avg_pool2d_gradient_matches_finite_difference(self) -> None:
        image = Tensor.from_list([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], requires_grad=True)
        weights = Tensor.from_list([
            [1.0, -2.0],
            [0.5, 3.0],
        ])
        loss = (avg_pool2d(image, (2, 2), stride=(1, 1)) * weights).sum()

        loss.backward()

        def loss_fn() -> float:
            pooled = avg_pool2d(image, (2, 2), stride=(1, 1))
            return (pooled * weights).sum()[0]

        self.assert_grad_close(image, loss_fn)

    def test_avg_pool2d_channel_gradient_matches_finite_difference(self) -> None:
        image = Tensor.from_list([
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            [
                [0.5, -1.0, 2.0],
                [3.0, 0.25, -2.0],
                [1.0, -0.75, 4.0],
            ],
        ], requires_grad=True)
        weights = Tensor.from_list([
            [
                [1.0, -2.0],
                [0.5, 3.0],
            ],
            [
                [-0.5, 1.5],
                [0.75, -2.0],
            ],
        ])
        loss = (avg_pool2d(image, (2, 2), stride=(1, 1)) * weights).sum()

        loss.backward()

        def loss_fn() -> float:
            pooled = avg_pool2d(image, (2, 2), stride=(1, 1))
            return (pooled * weights).sum()[0]

        self.assert_grad_close(image, loss_fn)

    def test_transpose_gradient_matches_finite_difference(self) -> None:
        x = Tensor.from_list([
            [1.0, -2.0, 3.0],
            [0.5, 4.0, -1.0],
        ], requires_grad=True)
        weights = Tensor.from_list([
            [2.0, -1.0],
            [0.5, 3.0],
        ])
        loss = matmul(x.T, weights).sum()

        loss.backward()

        def loss_fn() -> float:
            return matmul(x.T, weights).sum()[0]

        self.assert_grad_close(x, loss_fn)

    def test_permute_gradient_matches_finite_difference(self) -> None:
        x = Tensor.from_list([
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            [
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ],
        ], requires_grad=True)
        weights = Tensor.from_list([
            [
                [0.5, -1.0],
                [2.0, 3.0],
            ],
            [
                [1.5, -0.5],
                [0.25, 4.0],
            ],
            [
                [-2.0, 0.75],
                [1.25, -3.0],
            ],
        ])
        loss = (x.permute((2, 0, 1)) * weights).sum()

        loss.backward()

        def loss_fn() -> float:
            return (x.permute((2, 0, 1)) * weights).sum()[0]

        self.assert_grad_close(x, loss_fn)

    def test_axis_reduction_gradients(self) -> None:
        x = Tensor.from_list([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ], requires_grad=True)
        row_weights = Tensor.from_list([2.0, -1.0])
        col_weights = Tensor.from_list([0.5, 1.5, -2.0])
        loss = (x.sum(axis=1) * row_weights).sum()
        loss = loss + (x.mean(axis=0) * col_weights).sum()

        loss.backward()

        def loss_fn() -> float:
            row_loss = (x.sum(axis=1) * row_weights).sum()
            col_loss = (x.mean(axis=0) * col_weights).sum()
            return (row_loss + col_loss)[0]

        self.assert_grad_close(x, loss_fn)

    def test_nd_axis_reduction_gradients(self) -> None:
        x = Tensor.from_list([
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            [
                [5.0, 6.0],
                [7.0, 8.0],
            ],
        ], requires_grad=True)
        weights = Tensor.from_list([
            [0.5, -1.0],
            [2.0, 3.0],
        ])
        loss = (x.sum(axis=2) * weights).sum()
        loss = loss + x.mean(axis=0).sum()

        loss.backward()

        def loss_fn() -> float:
            weighted = (x.sum(axis=2) * weights).sum()
            return (weighted + x.mean(axis=0).sum())[0]

        self.assert_grad_close(x, loss_fn)

    def test_nd_broadcast_gradients_match_finite_difference(self) -> None:
        x = Tensor.from_list([
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            [
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ],
        ], requires_grad=True)
        row = Tensor.from_list([0.5, -1.0, 2.0], requires_grad=True)
        column = Tensor.from_list([
            [
                [1.0],
                [-2.0],
            ],
        ], requires_grad=True)
        loss = ((x * row) + column).sum()

        loss.backward()

        def loss_fn() -> float:
            return ((x * row) + column).sum()[0]

        self.assert_grad_close(x, loss_fn)
        self.assert_grad_close(row, loss_fn)
        self.assert_grad_close(column, loss_fn)

    def test_row_broadcast_bias_gradient(self) -> None:
        inputs = Tensor.from_list([
            [1.0, 2.0],
            [3.0, 4.0],
        ])
        bias = Tensor.from_list([10.0, 20.0], requires_grad=True)

        loss = (inputs + bias).sum()
        loss.backward()

        self.assertEqual(bias.grad, [2.0, 2.0])

    def test_scalar_tensor_broadcast_gradient(self) -> None:
        inputs = Tensor.from_list([
            [1.0, 2.0],
            [3.0, 4.0],
        ], requires_grad=True)
        bias = Tensor.from_list([10.0], requires_grad=True)
        scale = Tensor.from_list([0.5], requires_grad=True)
        loss = ((inputs + bias) * scale).sum()

        loss.backward()

        self.assertEqual(inputs.grad, [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(bias.grad, [2.0])
        self.assertEqual(scale.grad, [50.0])

    def test_row_broadcast_multiply_gradient(self) -> None:
        inputs = Tensor.from_list([
            [1.0, 2.0],
            [3.0, 4.0],
        ], requires_grad=True)
        row = Tensor.from_list([10.0, -2.0], requires_grad=True)
        loss = (inputs * row).sum()

        loss.backward()

        self.assertEqual(inputs.grad, [10.0, -2.0, 10.0, -2.0])
        self.assertEqual(row.grad, [4.0, 6.0])

    def test_division_gradient_matches_finite_difference(self) -> None:
        inputs = Tensor.from_list([
            [2.0, 4.0],
            [6.0, 8.0],
        ], requires_grad=True)
        row = Tensor.from_list([0.5, 2.0], requires_grad=True)
        loss = ((inputs / row) + (12 / inputs)).sum()

        loss.backward()

        def loss_fn() -> float:
            return ((inputs / row) + (12 / inputs)).sum()[0]

        self.assert_grad_close(inputs, loss_fn)
        self.assert_grad_close(row, loss_fn)

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
                model = TensorLinear(
                    inputs=2,
                    outputs=1,
                    weight=Tensor.zeros((1, 2), requires_grad=True),
                    bias=Tensor.zeros((1,), requires_grad=True),
                )

                summary = train_tensor_binary_classifier(
                    lambda: model(inputs).sigmoid(),
                    targets,
                    model.parameters(),
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
        model = TensorMLP(inputs=2, layers=[2, 1])
        model.layers[0].weight = Tensor.from_list([
            [1.0, -1.0],
            [-1.0, 1.0],
        ], requires_grad=True)
        model.layers[0].bias = Tensor.from_list([0.0, 0.0], requires_grad=True)
        model.layers[1].weight = Tensor.from_list([[0.1, 0.1]], requires_grad=True)
        model.layers[1].bias = Tensor.from_list([0.0], requires_grad=True)

        summary = train_tensor_binary_classifier(
            lambda: model(inputs).sigmoid(),
            targets,
            model.parameters(),
            steps=2000,
            lr=0.5,
        )

        self.assertLess(summary.final_loss, summary.initial_loss)
        self.assertLess(summary.final_loss, 0.01)
        self.assertEqual(summary.accuracy, 1.0)

    def test_tensor_linear_learns_tiny_multiclass_clusters(self) -> None:
        inputs = Tensor.from_list([
            [2.0, 0.0],
            [3.0, 0.0],
            [0.0, 2.0],
            [0.0, 3.0],
            [-2.0, -2.0],
            [-3.0, -2.0],
        ])
        targets = Tensor.from_list([0, 0, 1, 1, 2, 2])
        model = TensorLinear(
            inputs=2,
            outputs=3,
            weight=Tensor.zeros((3, 2), requires_grad=True),
            bias=Tensor.zeros((3,), requires_grad=True),
        )

        summary = train_tensor_multiclass_classifier(
            lambda: model(inputs),
            targets,
            model.parameters(),
            steps=500,
            lr=0.2,
        )

        self.assertLess(summary.final_loss, summary.initial_loss)
        self.assertLess(summary.final_loss, 0.01)
        self.assertEqual(summary.accuracy, 1.0)

    def test_tensor_multiclass_dataset_training_uses_batches(self) -> None:
        dataset = TinyDataset(
            xs=[
                [2.0, 0.0],
                [3.0, 0.0],
                [0.0, 2.0],
                [0.0, 3.0],
                [-2.0, -2.0],
                [-3.0, -2.0],
            ],
            ys=[
                0,
                0,
                1,
                1,
                2,
                2,
            ],
        )
        validation_dataset = TinyDataset(
            xs=[
                [2.5, 0.0],
                [0.0, 2.5],
                [-2.5, -2.0],
            ],
            ys=[
                0,
                1,
                2,
            ],
        )
        model = TensorLinear(
            inputs=2,
            outputs=3,
            weight=Tensor.zeros((3, 2), requires_grad=True),
            bias=Tensor.zeros((3,), requires_grad=True),
        )

        summary = train_tensor_multiclass_dataset(
            model,
            dataset,
            validation_dataset=validation_dataset,
            epochs=100,
            batch_size=2,
            lr=0.2,
            shuffle=True,
            seed=0,
        )

        self.assertLess(summary.final_loss, summary.initial_loss)
        self.assertEqual(summary.accuracy, 1.0)
        self.assertEqual(summary.validation_accuracy, 1.0)
        self.assertIsNotNone(summary.evaluation_loss)
        self.assertIsNotNone(summary.validation_loss)
        self.assertGreater(summary.evaluation_loss, 0.0)
        self.assertGreater(summary.validation_loss, 0.0)

    def test_tensor_multiclass_dataset_evaluation_uses_batches(self) -> None:
        dataset = TinyDataset(
            xs=[
                [2.0, 0.0],
                [0.0, 2.0],
                [-2.0, -2.0],
            ],
            ys=[
                0,
                1,
                2,
            ],
        )
        model = TensorLinear(
            inputs=2,
            outputs=3,
            weight=Tensor.from_list(
                [
                    [2.0, 0.0],
                    [0.0, 2.0],
                    [-2.0, -2.0],
                ],
            ),
            bias=Tensor.zeros((3,)),
        )

        summary = evaluate_tensor_multiclass_dataset(
            model,
            dataset,
            batch_size=2,
        )

        self.assertGreater(summary.loss, 0.0)
        self.assertEqual(summary.accuracy, 1.0)
        self.assertGreaterEqual(summary.elapsed_seconds, 0.0)

    def test_tensor_multiclass_dataset_evaluation_batch_error(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_tensor_multiclass_dataset(
                TensorLinear(inputs=1, outputs=2),
                TinyDataset([[0.0]], [0.0]),
                batch_size=0,
            )

    def test_tensor_multiclass_dataset_training_epoch_callback(self) -> None:
        dataset = TinyDataset(
            xs=[
                [2.0, 0.0],
                [0.0, 2.0],
            ],
            ys=[
                0,
                1,
            ],
        )
        model = TensorLinear(
            inputs=2,
            outputs=2,
            weight=Tensor.zeros((2, 2), requires_grad=True),
            bias=Tensor.zeros((2,), requires_grad=True),
        )
        reports = []

        train_tensor_multiclass_dataset(
            model,
            dataset,
            epochs=3,
            batch_size=1,
            lr=0.2,
            shuffle=False,
            epoch_callback=lambda epoch, summary: reports.append(
                (epoch, summary.evaluation_loss, summary.accuracy)
            ),
        )

        self.assertEqual([report[0] for report in reports], [1, 2, 3])
        self.assertTrue(all(report[1] > 0.0 for report in reports))
        self.assertTrue(all(report[2] is not None for report in reports))

    def test_tensor_multiclass_dataset_training_epoch_error(self) -> None:
        with self.assertRaises(ValueError):
            train_tensor_multiclass_dataset(
                TensorLinear(inputs=1, outputs=2),
                TinyDataset([[0.0]], [0.0]),
                epochs=0,
            )

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
