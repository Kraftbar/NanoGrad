import math
import unittest

from tensor import (
    Tensor,
    avg_pool2d,
    conv2d_valid,
    matmul,
    reshape,
    tensor_exp,
    tensor_log,
    tensor_mean,
    tensor_reciprocal,
    tensor_relu,
    tensor_sigmoid,
    tensor_sum,
    transpose,
)


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

    def test_nd_tensor_shape_indexing_and_list_conversion(self) -> None:
        x = Tensor.from_list([
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            [
                [7, 8, 9],
                [10, 11, 12],
            ],
        ])

        self.assertEqual(x.shape, (2, 2, 3))
        self.assertEqual(x[0, 0, 0], 1.0)
        self.assertEqual(x[1, 1, 2], 12.0)
        self.assertEqual(x[-1, -1, -1], 12.0)
        self.assertEqual(
            x.tolist(),
            [
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ],
                [
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                ],
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
        self.assertEqual(
            Tensor.zeros((2, 1, 2)).tolist(),
            [
                [
                    [0.0, 0.0],
                ],
                [
                    [0.0, 0.0],
                ],
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

    def test_nd_elementwise_add_and_multiply(self) -> None:
        a = Tensor.from_list([
            [
                [1, 2],
                [3, 4],
            ],
            [
                [5, 6],
                [7, 8],
            ],
        ])
        b = Tensor.from_list([
            [
                [10, 20],
                [30, 40],
            ],
            [
                [50, 60],
                [70, 80],
            ],
        ])

        self.assertEqual(
            (a + b).tolist(),
            [
                [
                    [11.0, 22.0],
                    [33.0, 44.0],
                ],
                [
                    [55.0, 66.0],
                    [77.0, 88.0],
                ],
            ],
        )
        self.assertEqual(
            (a * Tensor.from_list([2])).tolist(),
            [
                [
                    [2.0, 4.0],
                    [6.0, 8.0],
                ],
                [
                    [10.0, 12.0],
                    [14.0, 16.0],
                ],
            ],
        )

    def test_scalar_broadcast_and_subtraction(self) -> None:
        x = Tensor.from_list([1, 2, 3])
        scalar = Tensor.from_list([2])

        self.assertEqual((x + 10).tolist(), [11.0, 12.0, 13.0])
        self.assertEqual((10 + x).tolist(), [11.0, 12.0, 13.0])
        self.assertEqual((x - 1).tolist(), [0.0, 1.0, 2.0])
        self.assertEqual((10 - x).tolist(), [9.0, 8.0, 7.0])
        self.assertEqual((x / 2).tolist(), [0.5, 1.0, 1.5])
        self.assertEqual((12 / x).tolist(), [12.0, 6.0, 4.0])
        self.assertEqual((x + scalar).tolist(), [3.0, 4.0, 5.0])
        self.assertEqual((scalar + x).tolist(), [3.0, 4.0, 5.0])
        self.assertEqual((x * scalar).tolist(), [2.0, 4.0, 6.0])
        self.assertEqual((scalar * x).tolist(), [2.0, 4.0, 6.0])

    def test_scalar_tensor_broadcast_to_matrix(self) -> None:
        x = Tensor.from_list([
            [1, 2],
            [3, 4],
        ])
        scalar = Tensor.from_list([10])

        self.assertEqual(
            (x + scalar).tolist(),
            [
                [11.0, 12.0],
                [13.0, 14.0],
            ],
        )
        self.assertEqual(
            (scalar * x).tolist(),
            [
                [10.0, 20.0],
                [30.0, 40.0],
            ],
        )

    def test_row_vector_broadcast(self) -> None:
        x = Tensor.from_list([
            [1, 2, 3],
            [4, 5, 6],
        ])
        row = Tensor.from_list([10, 20, 30])

        self.assertEqual(
            (x + row).tolist(),
            [
                [11.0, 22.0, 33.0],
                [14.0, 25.0, 36.0],
            ],
        )
        self.assertEqual((row + x).tolist(), (x + row).tolist())
        self.assertEqual(
            (x - row).tolist(),
            [
                [-9.0, -18.0, -27.0],
                [-6.0, -15.0, -24.0],
            ],
        )
        self.assertEqual(
            (row - x).tolist(),
            [
                [9.0, 18.0, 27.0],
                [6.0, 15.0, 24.0],
            ],
        )
        self.assertEqual(
            (x * row).tolist(),
            [
                [10.0, 40.0, 90.0],
                [40.0, 100.0, 180.0],
            ],
        )
        self.assertEqual(
            (x / row).tolist(),
            [
                [0.1, 0.1, 0.1],
                [0.4, 0.25, 0.2],
            ],
        )
        self.assertEqual(
            (row / x).tolist(),
            [
                [10.0, 10.0, 10.0],
                [2.5, 4.0, 5.0],
            ],
        )

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

    def test_vector_matrix_multiply(self) -> None:
        x = Tensor.from_list([10, 20, 30])
        a = Tensor.from_list([
            [1, 2],
            [3, 4],
            [5, 6],
        ])

        self.assertEqual(matmul(x, a).tolist(), [220.0, 280.0])

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

    def test_reshape(self) -> None:
        x = Tensor.from_list([1, 2, 3, 4, 5, 6])

        self.assertEqual(
            reshape(x, (2, 3)).tolist(),
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
        )
        self.assertEqual(x.reshape((3, 2)).shape, (3, 2))
        self.assertEqual(x.reshape((3, 2)).reshape((6,)).tolist(), x.tolist())
        self.assertEqual(
            x.reshape((1, 2, 3)).tolist(),
            [
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ],
            ],
        )

    def test_conv2d_valid(self) -> None:
        image = Tensor.from_list([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ])
        edge_kernel = Tensor.from_list([
            [1, 0],
            [0, -1],
        ])
        sum_kernel = Tensor.from_list([
            [1, 1],
            [1, 1],
        ])

        self.assertEqual(
            conv2d_valid(image, edge_kernel).tolist(),
            [
                [-4.0, -4.0],
                [-4.0, -4.0],
            ],
        )
        self.assertEqual(
            image.conv2d_valid(sum_kernel).tolist(),
            [
                [12.0, 16.0],
                [24.0, 28.0],
            ],
        )

    def test_avg_pool2d(self) -> None:
        image = Tensor.from_list([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ])

        self.assertEqual(
            avg_pool2d(image, (2, 2)).tolist(),
            [
                [3.5, 5.5],
                [11.5, 13.5],
            ],
        )
        self.assertEqual(
            image.avg_pool2d((2, 2), stride=(1, 2)).tolist(),
            [
                [3.5, 5.5],
                [7.5, 9.5],
                [11.5, 13.5],
            ],
        )

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

    def test_nd_sum_and_mean_reductions(self) -> None:
        x = Tensor.from_list([
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            [
                [7, 8, 9],
                [10, 11, 12],
            ],
        ])

        self.assertEqual(
            x.sum(axis=0).tolist(),
            [
                [8.0, 10.0, 12.0],
                [14.0, 16.0, 18.0],
            ],
        )
        self.assertEqual(
            x.sum(axis=1).tolist(),
            [
                [5.0, 7.0, 9.0],
                [17.0, 19.0, 21.0],
            ],
        )
        self.assertEqual(
            x.sum(axis=2).tolist(),
            [
                [6.0, 15.0],
                [24.0, 33.0],
            ],
        )
        self.assertEqual(x.sum(axis=-1).tolist(), x.sum(axis=2).tolist())
        self.assertEqual(
            x.mean(axis=2).tolist(),
            [
                [2.0, 5.0],
                [8.0, 11.0],
            ],
        )

    def test_1d_sum_and_mean(self) -> None:
        x = Tensor.from_list([1, 2, 3])

        self.assertEqual(x.sum().tolist(), [6.0])
        self.assertEqual(x.sum(axis=0).tolist(), [6.0])
        self.assertEqual(x.mean().tolist(), [2.0])
        self.assertEqual(x.mean(axis=0).tolist(), [2.0])

    def test_exp_and_log(self) -> None:
        x = Tensor.from_list([
            [1.0, math.e],
            [math.e**2, math.e**3],
        ])

        self.assertEqual(tensor_log(x).shape, x.shape)
        self.assertAlmostEqual(x.log()[0, 0], 0.0)
        self.assertAlmostEqual(x.log()[0, 1], 1.0)
        self.assertAlmostEqual(x.log()[1, 0], 2.0)
        self.assertAlmostEqual(x.log()[1, 1], 3.0)
        self.assertAlmostEqual(tensor_exp(Tensor.from_list([0.0]))[0], 1.0)

    def test_reciprocal(self) -> None:
        x = Tensor.from_list([2.0, 4.0, -8.0])

        self.assertEqual(tensor_reciprocal(x).tolist(), [0.5, 0.25, -0.125])
        self.assertEqual(x.reciprocal().tolist(), tensor_reciprocal(x).tolist())

    def test_relu(self) -> None:
        x = Tensor.from_list([
            [-2, -1, 0],
            [1, 2, 3],
        ])

        self.assertEqual(
            tensor_relu(x).tolist(),
            [
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0],
            ],
        )
        self.assertEqual(x.relu().tolist(), tensor_relu(x).tolist())

    def test_sigmoid(self) -> None:
        x = Tensor.from_list([-2.0, 0.0, 2.0])
        y = tensor_sigmoid(x)

        self.assertEqual(y.shape, x.shape)
        self.assertAlmostEqual(y[0], 1 / (1 + math.exp(2.0)))
        self.assertAlmostEqual(y[1], 0.5)
        self.assertAlmostEqual(y[2], 1 / (1 + math.exp(-2.0)))
        self.assertEqual(x.sigmoid().tolist(), y.tolist())

    def test_shape_errors(self) -> None:
        with self.assertRaises(ValueError):
            Tensor.from_list([
                [1, 2],
                [3],
            ])

        with self.assertRaises(ValueError):
            Tensor.from_list([
                [
                    [1, 2],
                ],
                [
                    [3],
                ],
            ])

        with self.assertRaises(IndexError):
            Tensor.from_list([
                [
                    [1],
                ],
            ])[0, 0]

        with self.assertRaises(ValueError):
            Tensor([1], ())

        with self.assertRaises(ValueError):
            Tensor.from_list([1, 2]) + Tensor.from_list([1, 2, 3])

        with self.assertRaises(ValueError):
            Tensor.from_list([
                [1, 2],
                [3, 4],
            ]) + Tensor.from_list([1, 2, 3])

        with self.assertRaises(ValueError):
            matmul(Tensor.from_list([1, 2]), Tensor.from_list([1, 2, 3]))

        with self.assertRaises(ValueError):
            matmul(
                Tensor.from_list([1, 2]),
                Tensor.from_list([
                    [1, 2],
                    [3, 4],
                    [5, 6],
                ]),
            )

        with self.assertRaises(ValueError):
            Tensor.from_list([1, 2]).T

        with self.assertRaises(ValueError):
            Tensor.from_list([
                [1, 2],
                [3, 4],
            ]).sum(axis=2)

        with self.assertRaises(ValueError):
            Tensor.from_list([
                [
                    [1],
                ],
            ]).sum(axis=3)

        with self.assertRaises(ValueError):
            Tensor.from_list([1, 2, 3]).reshape((2, 2))

        with self.assertRaises(ValueError):
            conv2d_valid(Tensor.from_list([1, 2]), Tensor.from_list([[1]]))

        with self.assertRaises(ValueError):
            conv2d_valid(
                Tensor.from_list([[1, 2]]),
                Tensor.from_list([
                    [1, 2],
                    [3, 4],
                ]),
            )

        with self.assertRaises(ValueError):
            avg_pool2d(Tensor.from_list([1, 2]), (1, 1))

        with self.assertRaises(ValueError):
            avg_pool2d(Tensor.from_list([[1, 2]]), (0, 1))

        with self.assertRaises(ValueError):
            avg_pool2d(Tensor.from_list([[1, 2]]), (1, 1), stride=(0, 1))

        with self.assertRaises(ValueError):
            avg_pool2d(Tensor.from_list([[1, 2]]), (2, 1))

        with self.assertRaises(ValueError):
            Tensor.from_list([1, 0, -1]).log()

        with self.assertRaises(ValueError):
            Tensor.from_list([1, 0, -1]).reciprocal()

        with self.assertRaises(ValueError):
            Tensor.from_list([1, 2, 3]) / 0


if __name__ == "__main__":
    unittest.main()
