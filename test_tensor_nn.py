import unittest

from tensor import Tensor
from tensor_nn import binary_mlp, linear


class TensorNNTests(unittest.TestCase):
    def test_binary_mlp_vector_input(self) -> None:
        hidden_weight = Tensor.from_list([
            [1, -1],
            [-1, 1],
        ])
        hidden_bias = Tensor.from_list([0, 0])
        output_weight = Tensor.from_list([[10, 10]])
        output_bias = Tensor.from_list([-5])

        outputs = binary_mlp(
            Tensor.from_list([1, 0]),
            hidden_weight,
            hidden_bias,
            output_weight,
            output_bias,
        )

        self.assertEqual(outputs.shape, (1,))
        self.assertGreater(outputs[0], 0.99)

    def test_binary_mlp_batch_input(self) -> None:
        inputs = Tensor.from_list([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ])
        hidden_weight = Tensor.from_list([
            [1, -1],
            [-1, 1],
        ])
        hidden_bias = Tensor.from_list([0, 0])
        output_weight = Tensor.from_list([[10, 10]])
        output_bias = Tensor.from_list([-5])

        outputs = binary_mlp(
            inputs,
            hidden_weight,
            hidden_bias,
            output_weight,
            output_bias,
        )

        self.assertEqual(outputs.shape, (4, 1))
        self.assertLess(outputs[0, 0], 0.01)
        self.assertGreater(outputs[1, 0], 0.99)
        self.assertGreater(outputs[2, 0], 0.99)
        self.assertLess(outputs[3, 0], 0.01)

    def test_linear_vector_input(self) -> None:
        inputs = Tensor.from_list([10, 20, 30])
        weight = Tensor.from_list([
            [1, 2, 3],
            [4, 5, 6],
        ])
        bias = Tensor.from_list([1, -1])

        outputs = linear(inputs, weight, bias)

        self.assertEqual(outputs.shape, (2,))
        self.assertEqual(outputs.tolist(), [141.0, 319.0])

    def test_linear_batch_input(self) -> None:
        inputs = Tensor.from_list([
            [1, 2, 3],
            [10, 20, 30],
        ])
        weight = Tensor.from_list([
            [1, 2, 3],
            [4, 5, 6],
        ])
        bias = Tensor.from_list([1, -1])

        outputs = linear(inputs, weight, bias)

        self.assertEqual(outputs.shape, (2, 2))
        self.assertEqual(
            outputs.tolist(),
            [
                [15.0, 31.0],
                [141.0, 319.0],
            ],
        )

    def test_linear_shape_errors(self) -> None:
        with self.assertRaises(ValueError):
            linear(
                Tensor.from_list([1, 2]),
                Tensor.from_list([1, 2]),
                Tensor.from_list([0]),
            )

        with self.assertRaises(ValueError):
            linear(
                Tensor.from_list([1, 2]),
                Tensor.from_list([[1, 2]]),
                Tensor.from_list([[0]]),
            )

        with self.assertRaises(ValueError):
            linear(
                Tensor.from_list([[1, 2]]),
                Tensor.from_list([[1, 2], [3, 4]]),
                Tensor.from_list([0]),
            )


if __name__ == "__main__":
    unittest.main()
