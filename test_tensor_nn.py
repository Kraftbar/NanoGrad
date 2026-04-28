import unittest

from tensor import Tensor
from tensor_nn import linear


class TensorNNTests(unittest.TestCase):
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
