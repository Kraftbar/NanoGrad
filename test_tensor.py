import unittest

from tensor import Tensor, matmul, tensor_mean, tensor_sum, transpose


class TensorTests(unittest.TestCase):
    def test_1d_tensor_shape_indexing_and_list_conversion(self) -> None:
        x = Tensor.from_list([1, 2, 3])

        self.assertEqual(x.shape, (3,))
        self.assertEqual(x[0], 1.0)
        self.assertEqual(x[-1], 3.0)
        self.assertEqual(x.tolist(), [1.0, 2.0, 3.0])

    def test_2d_tensor_shape_indexing_and_list_conversion(self) -> None:
        x = Tensor.from_list([
            [1, 2, 3],
            [4, 5, 6],
        ])

        self.assertEqual(x.shape, (2, 3))
        self.assertEqual(x[0, 0], 1.0)
        self.assertEqual(x[1, 2], 6.0)
        self.assertEqual(
            x.tolist(),
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
        )

    def test_zeros(self) -> None:
        self.assertEqual(
            Tensor.zeros((2, 2)).tolist(),
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ],
        )

    def test_elementwise_add_and_multiply(self) -> None:
        a = Tensor.from_list([
            [1, 2],
            [3, 4],
        ])
        b = Tensor.from_list([
            [10, 20],
            [30, 40],
        ])

        self.assertEqual(
            (a + b).tolist(),
            [
                [11.0, 22.0],
                [33.0, 44.0],
            ],
        )
        self.assertEqual(
            (a * 2).tolist(),
            [
                [2.0, 4.0],
                [6.0, 8.0],
            ],
        )

    def test_scalar_broadcast_and_subtraction(self) -> None:
        x = Tensor.from_list([1, 2, 3])

        self.assertEqual((x + 10).tolist(), [11.0, 12.0, 13.0])
        self.assertEqual((10 + x).tolist(), [11.0, 12.0, 13.0])
        self.assertEqual((x - 1).tolist(), [0.0, 1.0, 2.0])
        self.assertEqual((10 - x).tolist(), [9.0, 8.0, 7.0])

    def test_vector_dot_product(self) -> None:
        a = Tensor.from_list([1, 2, 3])
        b = Tensor.from_list([4, 5, 6])

        self.assertEqual(matmul(a, b).tolist(), [32.0])

    def test_matrix_vector_multiply(self) -> None:
        a = Tensor.from_list([
            [1, 2, 3],
            [4, 5, 6],
        ])
        x = Tensor.from_list([10, 20, 30])

        self.assertEqual(matmul(a, x).tolist(), [140.0, 320.0])

    def test_matrix_matrix_multiply(self) -> None:
        a = Tensor.from_list([
            [1, 2, 3],
            [4, 5, 6],
        ])
        b = Tensor.from_list([
            [7, 8],
            [9, 10],
            [11, 12],
        ])

        self.assertEqual(
            matmul(a, b).tolist(),
            [
                [58.0, 64.0],
                [139.0, 154.0],
            ],
        )

    def test_transpose(self) -> None:
        x = Tensor.from_list([
            [1, 2, 3],
            [4, 5, 6],
        ])

        self.assertEqual(
            transpose(x).tolist(),
            [
                [1.0, 4.0],
                [2.0, 5.0],
                [3.0, 6.0],
            ],
        )
        self.assertEqual(x.T.tolist(), transpose(x).tolist())

    def test_sum_reductions(self) -> None:
        x = Tensor.from_list([
            [1, 2, 3],
            [4, 5, 6],
        ])

        self.assertEqual(tensor_sum(x).tolist(), [21.0])
        self.assertEqual(x.sum(axis=0).tolist(), [5.0, 7.0, 9.0])
        self.assertEqual(x.sum(axis=1).tolist(), [6.0, 15.0])

    def test_mean_reductions(self) -> None:
        x = Tensor.from_list([
            [1, 2, 3],
            [4, 5, 6],
        ])

        self.assertEqual(tensor_mean(x).tolist(), [3.5])
        self.assertEqual(x.mean(axis=0).tolist(), [2.5, 3.5, 4.5])
        self.assertEqual(x.mean(axis=1).tolist(), [2.0, 5.0])

    def test_1d_sum_and_mean(self) -> None:
        x = Tensor.from_list([1, 2, 3])

        self.assertEqual(x.sum().tolist(), [6.0])
        self.assertEqual(x.sum(axis=0).tolist(), [6.0])
        self.assertEqual(x.mean().tolist(), [2.0])
        self.assertEqual(x.mean(axis=0).tolist(), [2.0])

    def test_shape_errors(self) -> None:
        with self.assertRaises(ValueError):
            Tensor.from_list([
                [1, 2],
                [3],
            ])

        with self.assertRaises(ValueError):
            Tensor.from_list([1, 2]) + Tensor.from_list([1, 2, 3])

        with self.assertRaises(ValueError):
            matmul(Tensor.from_list([1, 2]), Tensor.from_list([1, 2, 3]))

        with self.assertRaises(ValueError):
            Tensor.from_list([1, 2]).T

        with self.assertRaises(ValueError):
            Tensor.from_list([
                [1, 2],
                [3, 4],
            ]).sum(axis=2)


if __name__ == "__main__":
    unittest.main()
