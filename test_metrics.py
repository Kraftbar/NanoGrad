import unittest

from metrics import (
    tensor_binary_accuracy,
    tensor_multiclass_accuracy,
    tensor_multiclass_confusion_matrix,
)
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

    def test_tensor_multiclass_accuracy(self) -> None:
        accuracy = tensor_multiclass_accuracy(
            logits=Tensor.from_list([
                [5.0, 1.0, 0.0],
                [1.0, 2.0, 4.0],
                [0.0, 3.0, 1.0],
            ]),
            targets=[0, 1, 1],
        )

        self.assertEqual(accuracy, 2 / 3)

    def test_tensor_multiclass_accuracy_accepts_column_targets(self) -> None:
        accuracy = tensor_multiclass_accuracy(
            logits=Tensor.from_list([
                [5.0, 1.0],
                [1.0, 4.0],
            ]),
            targets=Tensor.from_list([
                [0],
                [1],
            ]),
        )

        self.assertEqual(accuracy, 1.0)

    def test_tensor_multiclass_confusion_matrix(self) -> None:
        matrix = tensor_multiclass_confusion_matrix(
            logits=Tensor.from_list([
                [5.0, 1.0, 0.0],
                [1.0, 2.0, 4.0],
                [0.0, 3.0, 1.0],
                [0.0, 2.0, 4.0],
            ]),
            targets=[0, 1, 1, 2],
        )

        self.assertEqual(
            matrix,
            [
                [1, 0, 0],
                [0, 1, 1],
                [0, 0, 1],
            ],
        )

    def test_tensor_multiclass_confusion_matrix_accepts_column_targets(self) -> None:
        matrix = tensor_multiclass_confusion_matrix(
            logits=Tensor.from_list([
                [5.0, 1.0],
                [1.0, 4.0],
            ]),
            targets=Tensor.from_list([
                [0],
                [1],
            ]),
        )

        self.assertEqual(matrix, [[1, 0], [0, 1]])

    def test_tensor_multiclass_accuracy_errors(self) -> None:
        with self.assertRaises(ValueError):
            tensor_multiclass_accuracy(Tensor.from_list([1, 2]), [0])

        with self.assertRaises(ValueError):
            tensor_multiclass_accuracy(Tensor.from_list([[1, 2]]), [0, 1])

        with self.assertRaises(ValueError):
            tensor_multiclass_accuracy(Tensor.from_list([[1, 2]]), [2])

    def test_tensor_multiclass_confusion_matrix_errors(self) -> None:
        with self.assertRaises(ValueError):
            tensor_multiclass_confusion_matrix(Tensor.from_list([1, 2]), [0])

        with self.assertRaises(ValueError):
            tensor_multiclass_confusion_matrix(Tensor.from_list([[1, 2]]), [0, 1])

        with self.assertRaises(ValueError):
            tensor_multiclass_confusion_matrix(Tensor.from_list([[1, 2]]), [2])


if __name__ == "__main__":
    unittest.main()
