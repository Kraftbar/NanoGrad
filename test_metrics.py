import unittest

from metrics import tensor_binary_accuracy
from tensor import Tensor


class TensorMetricTests(unittest.TestCase):
    def test_tensor_binary_accuracy_perfect(self) -> None:
        accuracy = tensor_binary_accuracy(
            probabilities=Tensor.from_list([0.1, 0.8, 0.2, 0.9]),
            targets=Tensor.from_list([0, 1, 0, 1]),
        )

        self.assertEqual(accuracy, 1.0)

    def test_tensor_binary_accuracy_partial(self) -> None:
        accuracy = tensor_binary_accuracy(
            probabilities=Tensor.from_list([
                [0.1, 0.8],
                [0.4, 0.7],
            ]),
            targets=Tensor.from_list([
                [0, 1],
                [1, 1],
            ]),
        )

        self.assertEqual(accuracy, 0.75)

    def test_tensor_binary_accuracy_shape_error(self) -> None:
        with self.assertRaises(ValueError):
            tensor_binary_accuracy(
                probabilities=Tensor.from_list([0.5, 0.5]),
                targets=Tensor.from_list([0, 1, 1]),
            )

    def test_tensor_binary_accuracy_target_error(self) -> None:
        with self.assertRaises(ValueError):
            tensor_binary_accuracy(
                probabilities=Tensor.from_list([0.5, 0.5]),
                targets=Tensor.from_list([0, 0.25]),
            )


if __name__ == "__main__":
    unittest.main()
