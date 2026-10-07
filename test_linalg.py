import math
import random
import unittest

from linalg import (
    cross,
    determinant3,
    identity,
    inverse,
    mat_mul,
    mat_vec,
    nearest_rotation,
    norm,
    null_vector,
    solve,
    svd3,
    symmetric_eigen,
    transpose,
)


def random_matrix(rows, cols, seed):
    rng = random.Random(seed)
    return [[rng.uniform(-1.0, 1.0) for _ in range(cols)] for _ in range(rows)]


class LinalgTests(unittest.TestCase):
    def assertMatrixAlmostEqual(self, left, right, places=9) -> None:
        for left_row, right_row in zip(left, right):
            for a, b in zip(left_row, right_row):
                self.assertAlmostEqual(a, b, places=places)

    def test_basic_products(self) -> None:
        a = [[1.0, 2.0], [3.0, 4.0]]

        self.assertEqual(mat_mul(a, identity(2)), a)
        self.assertEqual(transpose(a), [[1.0, 3.0], [2.0, 4.0]])
        self.assertEqual(mat_vec(a, [1.0, -1.0]), [-1.0, -1.0])
        self.assertEqual(cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), [0.0, 0.0, 1.0])
        self.assertEqual(determinant3(identity(3)), 1.0)

    def test_solve(self) -> None:
        a = random_matrix(5, 5, seed=1)
        x = [1.0, -2.0, 0.5, 3.0, 0.0]

        solution = solve(a, mat_vec(a, x))

        for expected, actual in zip(x, solution):
            self.assertAlmostEqual(expected, actual)
        with self.assertRaises(ValueError):
            solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])

    def test_inverse(self) -> None:
        a = random_matrix(4, 4, seed=6)

        self.assertMatrixAlmostEqual(mat_mul(a, inverse(a)), identity(4))

    def test_symmetric_eigen_reconstructs_matrix(self) -> None:
        b = random_matrix(6, 6, seed=2)
        a = mat_mul(transpose(b), b)

        values, vectors = symmetric_eigen(a)

        self.assertEqual(values, sorted(values))
        for value, vector in zip(values, vectors):
            self.assertAlmostEqual(norm(vector), 1.0)
            for got, expected in zip(mat_vec(a, vector), vector):
                self.assertAlmostEqual(got, value * expected)

    def test_null_vector_of_rank_deficient_matrix(self) -> None:
        a = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 0.0, 1.0]]

        x = null_vector(a)

        self.assertAlmostEqual(norm(x), 1.0)
        for value in mat_vec(a, x):
            self.assertAlmostEqual(value, 0.0)

    def test_svd3_reconstructs_full_and_rank_two_matrices(self) -> None:
        full = random_matrix(3, 3, seed=3)
        rank_two = mat_mul(random_matrix(3, 2, seed=4), random_matrix(2, 3, seed=5))

        for matrix in (full, rank_two):
            u, singular, vt = svd3(matrix)
            rebuilt = mat_mul(
                mat_mul(u, [[singular[i] if i == j else 0.0 for j in range(3)] for i in range(3)]),
                vt,
            )
            self.assertMatrixAlmostEqual(rebuilt, matrix)
            self.assertMatrixAlmostEqual(mat_mul(transpose(u), u), identity(3))
            self.assertGreaterEqual(singular[0], singular[1])
            self.assertGreaterEqual(singular[1], singular[2])

        self.assertAlmostEqual(svd3(rank_two)[1][2], 0.0)

    def test_nearest_rotation(self) -> None:
        angle = 0.3
        rotation = [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
        noisy = [[value * 2.0 + 0.001 for value in row] for row in rotation]

        snapped = nearest_rotation(noisy)

        self.assertAlmostEqual(determinant3(snapped), 1.0)
        self.assertMatrixAlmostEqual(snapped, rotation, places=3)


if __name__ == "__main__":
    unittest.main()
