"""Small dense linear algebra on lists of lists, for geometry experiments.

Matrices are row lists: ``A[row][col]``. Everything is O(n^3) and meant for
the 3x3 .. 12x12 systems that show up in camera geometry, not for big data.

The two workhorses:

- ``solve``: Gaussian elimination with partial pivoting, for normal equations.
- ``symmetric_eigen``: the Jacobi rotation method. It repeatedly zeroes the
  largest off-diagonal entry with a plane rotation until the matrix is
  diagonal. Slow-ish but short, robust, and easy to follow.

Everything else (null vectors, a 3x3 SVD, nearest rotation) is built on those.
"""

from __future__ import annotations

import math


Vector = list[float]
Matrix = list[list[float]]


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


def identity(size: int) -> Matrix:
    return [
        [1.0 if row == col else 0.0 for col in range(size)]
        for row in range(size)
    ]


def transpose(matrix: Matrix) -> Matrix:
    return [list(column) for column in zip(*matrix)]


def mat_mul(left: Matrix, right: Matrix) -> Matrix:
    columns = transpose(right)
    return [
        [dot(row, column) for column in columns]
        for row in left
    ]


def mat_vec(matrix: Matrix, vector: Vector) -> Vector:
    return [dot(row, vector) for row in matrix]


def dot(left: Vector, right: Vector) -> float:
    return sum(a * b for a, b in zip(left, right))


def norm(vector: Vector) -> float:
    return math.sqrt(dot(vector, vector))


def normalize(vector: Vector) -> Vector:
    length = norm(vector)
    if length == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return [value / length for value in vector]


def cross(left: Vector, right: Vector) -> Vector:
    ax, ay, az = left
    bx, by, bz = right
    return [
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx,
    ]


def determinant3(matrix: Matrix) -> float:
    (a, b, c), (d, e, f), (g, h, i) = matrix
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def column(matrix: Matrix, index: int) -> Vector:
    return [row[index] for row in matrix]


def from_columns(columns: list[Vector]) -> Matrix:
    return transpose(columns)


# ---------------------------------------------------------------------------
# Solving and decompositions
# ---------------------------------------------------------------------------


def solve(matrix: Matrix, rhs: Vector) -> Vector:
    """Solve ``matrix @ x = rhs`` by Gaussian elimination with pivoting."""

    size = len(matrix)
    # Augmented matrix [A | b], copied so the caller's data is untouched.
    rows = [list(row) + [value] for row, value in zip(matrix, rhs)]

    for pivot in range(size):
        best = max(range(pivot, size), key=lambda row: abs(rows[row][pivot]))
        if abs(rows[best][pivot]) < 1e-300:
            raise ValueError("matrix is singular")
        rows[pivot], rows[best] = rows[best], rows[pivot]
        for row in range(pivot + 1, size):
            factor = rows[row][pivot] / rows[pivot][pivot]
            rows[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(rows[row], rows[pivot])
            ]

    solution = [0.0] * size
    for row in reversed(range(size)):
        known = sum(rows[row][col] * solution[col] for col in range(row + 1, size))
        solution[row] = (rows[row][size] - known) / rows[row][row]
    return solution


def symmetric_eigen(
    matrix: Matrix,
    *,
    tolerance: float = 1e-14,
    max_sweeps: int = 100,
) -> tuple[Vector, list[Vector]]:
    """Eigenvalues (ascending) and unit eigenvectors of a symmetric matrix.

    Jacobi method: for each off-diagonal pair (p, q), pick the plane rotation
    J that makes ``J^T A J`` zero at (p, q). Rotations keep the eigenvalues,
    so once A is diagonal its diagonal holds them, and the product of all
    rotations holds the eigenvectors as columns.
    """

    size = len(matrix)
    a = [list(row) for row in matrix]
    vectors = identity(size)
    scale = max(abs(value) for row in a for value in row) or 1.0

    for _ in range(max_sweeps):
        off_diagonal = sum(
            a[p][q] * a[p][q]
            for p in range(size)
            for q in range(size)
            if p != q
        )
        if off_diagonal <= (tolerance * scale) ** 2:
            break

        for p in range(size - 1):
            for q in range(p + 1, size):
                if a[p][q] == 0.0:
                    continue
                # Rotation angle that zeroes a[p][q]: tan(2 theta) = 2 a_pq / (a_qq - a_pp)
                theta = 0.5 * math.atan2(2.0 * a[p][q], a[q][q] - a[p][p])
                c = math.cos(theta)
                s = math.sin(theta)
                _rotate(a, vectors, p, q, c, s)

    values = [a[index][index] for index in range(size)]
    order = sorted(range(size), key=lambda index: values[index])
    return (
        [values[index] for index in order],
        [column(vectors, index) for index in order],
    )


def _rotate(a: Matrix, vectors: Matrix, p: int, q: int, c: float, s: float) -> None:
    """Apply A <- J^T A J and V <- V J for the (p, q) plane rotation J."""

    size = len(a)
    for k in range(size):
        a_kp = a[k][p]
        a_kq = a[k][q]
        a[k][p] = c * a_kp - s * a_kq
        a[k][q] = s * a_kp + c * a_kq
    for k in range(size):
        a_pk = a[p][k]
        a_qk = a[q][k]
        a[p][k] = c * a_pk - s * a_qk
        a[q][k] = s * a_pk + c * a_qk
    for k in range(size):
        v_kp = vectors[k][p]
        v_kq = vectors[k][q]
        vectors[k][p] = c * v_kp - s * v_kq
        vectors[k][q] = s * v_kp + c * v_kq


def null_vector(matrix: Matrix) -> Vector:
    """Unit x minimizing |A x| (the least-squares solution of A x = 0).

    |A x|^2 = x^T (A^T A) x, which a unit x minimizes by being the eigenvector
    of A^T A with the smallest eigenvalue.
    """

    gram = mat_mul(transpose(matrix), matrix)
    _, vectors = symmetric_eigen(gram)
    return vectors[0]


def svd3(matrix: Matrix) -> tuple[Matrix, Vector, Matrix]:
    """Singular value decomposition of a 3x3 matrix: M = U diag(S) V^T.

    Built from the eigen decomposition of M^T M = V diag(S^2) V^T, then
    u_i = M v_i / s_i. A (near) zero third singular value is common in
    geometry (essential matrices), so u_3 is taken as u_1 x u_2 when needed.
    Singular values come out in descending order; U and V are rotations
    or reflections exactly as the data dictates.
    """

    values, vectors = symmetric_eigen(mat_mul(transpose(matrix), matrix))
    order = [2, 1, 0]
    singular = [math.sqrt(max(values[index], 0.0)) for index in order]
    v_columns = [vectors[index] for index in order]

    u_columns = []
    for index, (value, v_column) in enumerate(zip(singular, v_columns)):
        if value > 1e-12 * max(singular[0], 1e-300) and index < 2:
            u_columns.append([entry / value for entry in mat_vec(matrix, v_column)])
        elif index < 2:
            raise ValueError("svd3 expects rank >= 2")
        else:
            u_column = cross(u_columns[0], u_columns[1])
            if value > 1e-12 * singular[0]:
                # Full rank: keep the sign implied by M v_3.
                if dot(u_column, mat_vec(matrix, v_column)) < 0.0:
                    u_column = [-entry for entry in u_column]
            u_columns.append(u_column)

    return from_columns(u_columns), singular, transpose(from_columns(v_columns))


def nearest_rotation(matrix: Matrix) -> Matrix:
    """Closest rotation (det = +1) to a 3x3 matrix in the Frobenius norm."""

    u, _, vt = svd3(matrix)
    rotation = mat_mul(u, vt)
    if determinant3(rotation) < 0.0:
        u = [row[:2] + [-row[2]] for row in u]
        rotation = mat_mul(u, vt)
    return rotation
