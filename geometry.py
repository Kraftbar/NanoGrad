"""Camera geometry and pose estimation, from the pinhole model up.

Conventions (the same ones ORB-SLAM3 and most multi-view geometry texts use):

- A ``Pose`` maps world points into the camera frame: X_cam = R X_world + t.
- The camera looks down its +z axis; x goes right and y goes down in the image.
- "Normalized" image coordinates are pixels with the intrinsics removed:
  (x, y) = (X_cam / Z_cam, Y_cam / Z_cam).

Pinhole projection, written out:

    [u]   [fx  0  cx] [X/Z]
    [v] = [ 0 fy  cy] [Y/Z]
    [1]   [ 0  0   1] [ 1 ]

Estimators in this file (all on small synthetic problems, pure Python):

- ``triangulate_point``: two views + a match -> 3D point (linear DLT)
- ``estimate_pose_dlt``: 3D points + their pixels -> camera pose (linear PnP)
- ``essential_matrix`` + ``relative_pose_from_essential``: matches between two
  views -> relative rotation and translation direction (8-point algorithm)
- ``refine_pose``: Levenberg-Marquardt on reprojection error, with the
  Jacobian taken from NanoGrad's scalar autograd engine
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from engine import Value, gradients
from linalg import (
    Matrix,
    Vector,
    column,
    determinant3,
    identity,
    mat_mul,
    mat_vec,
    nearest_rotation,
    norm,
    null_vector,
    solve,
    svd3,
    transpose,
)


# ---------------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------------


def skew(vector: Sequence) -> list[list]:
    """[v]x, the matrix with [v]x @ w = v x w (cross product as a matrix)."""

    x, y, z = vector
    return [
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ]


def rotation_from_axis_angle(axis_angle: Sequence) -> list[list]:
    """Rodrigues' formula: rotate by |w| radians about the axis w / |w|.

        R = I + A [w]x + B [w]x^2,   A = sin(theta) / theta,
                                     B = (1 - cos(theta)) / theta^2

    Works on floats or on autograd ``Value`` objects. Near theta = 0, A and B
    switch to their Taylor series in theta^2, which keeps the gradient finite
    at exactly zero rotation (where pose refinement starts every step).
    """

    theta_sq = sum(component * component for component in axis_angle)
    theta_sq_value = theta_sq.data if isinstance(theta_sq, Value) else theta_sq
    if theta_sq_value < 1e-8:
        a = 1.0 - theta_sq / 6.0
        b = 0.5 - theta_sq / 24.0
    else:
        theta = _sqrt(theta_sq)
        a = _sin(theta) / theta
        b = (1.0 - _cos(theta)) / theta_sq

    w = skew(axis_angle)
    w_sq = _mat_mul_generic(w, w)
    return [
        [
            (1.0 if row == col else 0.0) + a * w[row][col] + b * w_sq[row][col]
            for col in range(3)
        ]
        for row in range(3)
    ]


def axis_angle_from_rotation(rotation: Matrix) -> Vector:
    """Inverse of Rodrigues: the rotation vector w with R = exp([w]x)."""

    cos_theta = (rotation[0][0] + rotation[1][1] + rotation[2][2] - 1.0) / 2.0
    theta = math.acos(max(-1.0, min(1.0, cos_theta)))
    # (R - R^T) / 2 = sin(theta) [axis]x
    vee = [
        (rotation[2][1] - rotation[1][2]) / 2.0,
        (rotation[0][2] - rotation[2][0]) / 2.0,
        (rotation[1][0] - rotation[0][1]) / 2.0,
    ]
    if theta < 1e-8:
        return vee
    if math.pi - theta < 1e-6:
        # sin(theta) ~ 0: read the axis from R + I = 2 axis axis^T instead.
        diagonal = [rotation[i][i] + 1.0 for i in range(3)]
        best = max(range(3), key=lambda i: diagonal[i])
        axis_column = [rotation[i][best] + (1.0 if i == best else 0.0) for i in range(3)]
        length = norm(axis_column)
        return [theta * value / length for value in axis_column]
    scale = theta / math.sin(theta)
    return [scale * value for value in vee]


def rotation_angle_degrees(left: Matrix, right: Matrix) -> float:
    """Angle of the rotation that takes ``right`` to ``left``.

    Uses atan2(sin, cos) rather than acos(cos): acos loses almost all
    precision for the tiny angles we care about when checking estimates.
    """

    d = mat_mul(left, transpose(right))
    cos_theta = (d[0][0] + d[1][1] + d[2][2] - 1.0) / 2.0
    sin_theta = norm([d[2][1] - d[1][2], d[0][2] - d[2][0], d[1][0] - d[0][1]]) / 2.0
    return math.degrees(math.atan2(sin_theta, cos_theta))


# ---------------------------------------------------------------------------
# Poses and cameras
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pose:
    """World-to-camera rigid transform: X_cam = R X_world + t."""

    rotation: Matrix
    translation: Vector

    @classmethod
    def identity(cls) -> "Pose":
        return cls(identity(3), [0.0, 0.0, 0.0])

    @classmethod
    def from_axis_angle(cls, axis_angle: Vector, translation: Vector) -> "Pose":
        return cls(rotation_from_axis_angle(axis_angle), list(translation))

    def apply(self, point: Vector) -> Vector:
        rotated = mat_vec(self.rotation, point)
        return [r + t for r, t in zip(rotated, self.translation)]

    def inverse(self) -> "Pose":
        # X_world = R^T X_cam - R^T t
        rotation_t = transpose(self.rotation)
        return Pose(rotation_t, [-value for value in mat_vec(rotation_t, self.translation)])

    def compose(self, other: "Pose") -> "Pose":
        """self after other: first apply ``other``, then ``self``."""

        return Pose(
            mat_mul(self.rotation, other.rotation),
            self.apply(other.translation),
        )

    @property
    def center(self) -> Vector:
        """Camera center in world coordinates: C = -R^T t."""

        return self.inverse().translation


@dataclass(frozen=True)
class PinholeCamera:
    """Intrinsics of an ideal pinhole camera (no lens distortion)."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def matrix(self) -> Matrix:
        return [
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ]

    def project(self, point_camera: Vector) -> Vector | None:
        """Pixel (u, v) of a camera-frame point, or None if behind the camera."""

        x, y, z = point_camera
        if z <= 0.0:
            return None
        return [self.fx * x / z + self.cx, self.fy * y / z + self.cy]

    def normalize(self, pixel: Vector) -> Vector:
        """Remove intrinsics: pixel -> (X/Z, Y/Z)."""

        u, v = pixel
        return [(u - self.cx) / self.fx, (v - self.cy) / self.fy]

    def contains(self, pixel: Vector | None) -> bool:
        return (
            pixel is not None
            and 0.0 <= pixel[0] < self.width
            and 0.0 <= pixel[1] < self.height
        )


def project_points(
    camera: PinholeCamera,
    pose: Pose,
    points: Sequence[Vector],
) -> list[Vector | None]:
    return [camera.project(pose.apply(point)) for point in points]


def reprojection_rmse(
    camera: PinholeCamera,
    pose: Pose,
    points: Sequence[Vector],
    pixels: Sequence[Vector],
) -> float:
    """Root-mean-square pixel distance between observed and projected points."""

    total = 0.0
    for point, pixel in zip(points, pixels):
        projected = camera.project(pose.apply(point))
        if projected is None:
            return math.inf
        total += (projected[0] - pixel[0]) ** 2 + (projected[1] - pixel[1]) ** 2
    return math.sqrt(total / len(points))


# ---------------------------------------------------------------------------
# Linear estimators
# ---------------------------------------------------------------------------


def triangulate_point(
    first: Pose,
    second: Pose,
    first_normalized: Vector,
    second_normalized: Vector,
) -> Vector:
    """Linear (DLT) triangulation of one match seen by two posed cameras.

    With P = [R | t] (3x4) and an observation (x, y), the projection equation
    x = (P_1 . X) / (P_3 . X) rearranges to two linear equations in the
    homogeneous point X:

        x * P_3 - P_1 = 0
        y * P_3 - P_2 = 0

    Two views give four equations; X is the null vector of that 4x4 system.
    """

    rows = []
    for pose, (x, y) in ((first, first_normalized), (second, second_normalized)):
        projection = _projection_matrix(pose)
        rows.append([x * p3 - p1 for p1, p3 in zip(projection[0], projection[2])])
        rows.append([y * p3 - p2 for p2, p3 in zip(projection[1], projection[2])])
    homogeneous = null_vector(rows)
    if abs(homogeneous[3]) < 1e-12:
        raise ValueError("point is at infinity (no parallax between the views)")
    return [value / homogeneous[3] for value in homogeneous[:3]]


def estimate_pose_dlt(points: Sequence[Vector], normalized: Sequence[Vector]) -> Pose:
    """Linear PnP: camera pose from >= 6 known 3D points and their projections.

    Unknown: the 12 entries of P = [R | t]. Each point gives the same two
    equations as triangulation, now linear in P instead of in X:

        [X^T  0^T  -x X^T] [p1]
        [0^T  X^T  -y X^T] [p2] = 0       (X homogeneous, 4 values)
                           [p3]

    The null vector gives P up to scale. Its left 3x3 block is s * R, so we
    divide out s and snap that block to the nearest true rotation.
    """

    if len(points) < 6:
        raise ValueError("estimate_pose_dlt needs at least 6 points")

    rows = []
    for point, (x, y) in zip(points, normalized):
        h = [*point, 1.0]
        rows.append([*h, 0.0, 0.0, 0.0, 0.0, *[-x * value for value in h]])
        rows.append([0.0, 0.0, 0.0, 0.0, *h, *[-y * value for value in h]])
    p = null_vector(rows)
    projection = [p[0:4], p[4:8], p[8:12]]

    block = [row[:3] for row in projection]
    if determinant3(block) < 0.0:
        # The null vector's sign is arbitrary; R must have det +1.
        projection = [[-value for value in row] for row in projection]
        block = [row[:3] for row in projection]

    _, singular, _ = svd3(block)
    scale = sum(singular) / 3.0
    rotation = nearest_rotation(block)
    translation = [row[3] / scale for row in projection]
    return Pose(rotation, translation)


def essential_matrix(
    first_normalized: Sequence[Vector],
    second_normalized: Sequence[Vector],
) -> Matrix:
    """Eight-point algorithm for the essential matrix E = [t]x R.

    A match (x1, x2) in normalized coordinates satisfies the epipolar
    constraint x2^T E x1 = 0, which is one linear equation in the 9 entries
    of E. Eight or more matches pin E down up to scale. Then E is projected
    onto valid essential matrices: two equal singular values and one zero.
    """

    if len(first_normalized) < 8:
        raise ValueError("essential_matrix needs at least 8 matches")

    rows = []
    for (x1, y1), (x2, y2) in zip(first_normalized, second_normalized):
        rows.append([
            x2 * x1, x2 * y1, x2,
            y2 * x1, y2 * y1, y2,
            x1, y1, 1.0,
        ])
    e = null_vector(rows)
    estimate = [e[0:3], e[3:6], e[6:9]]

    u, singular, vt = svd3(estimate)
    sigma = (singular[0] + singular[1]) / 2.0
    return mat_mul(mat_mul(u, [[sigma, 0, 0], [0, sigma, 0], [0, 0, 0]]), vt)


def relative_pose_from_essential(
    essential: Matrix,
    first_normalized: Sequence[Vector],
    second_normalized: Sequence[Vector],
) -> Pose:
    """Recover (R, t) of the second camera relative to the first.

    E = U diag(1, 1, 0) V^T has four (R, t) factorizations:

        R in {U W V^T, U W^T V^T},  t = +/- u_3,
        W = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]

    Only one puts the triangulated points in front of both cameras (the
    "cheirality" check). ``t`` has unit length: two views alone cannot tell
    a small nearby scene from a big faraway one.
    """

    u, _, vt = svd3(essential)
    if determinant3(u) < 0.0:
        u = [[-value for value in row] for row in u]
    if determinant3(vt) < 0.0:
        vt = [[-value for value in row] for row in vt]

    w = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    rotations = [mat_mul(mat_mul(u, w), vt), mat_mul(mat_mul(u, transpose(w)), vt)]
    u3 = column(u, 2)
    candidates = [
        Pose(rotation, [sign * value for value in u3])
        for rotation in rotations
        for sign in (1.0, -1.0)
    ]

    first = Pose.identity()

    def points_in_front(candidate: Pose) -> int:
        count = 0
        for n1, n2 in zip(first_normalized, second_normalized):
            point = triangulate_point(first, candidate, n1, n2)
            if point[2] > 0.0 and candidate.apply(point)[2] > 0.0:
                count += 1
        return count

    return max(candidates, key=points_in_front)


# ---------------------------------------------------------------------------
# Nonlinear refinement with autograd
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RefinementStep:
    iteration: int
    rmse: float
    damping: float


def refine_pose(
    camera: PinholeCamera,
    pose: Pose,
    points: Sequence[Vector],
    pixels: Sequence[Vector],
    *,
    iterations: int = 10,
    damping: float = 1e-3,
) -> tuple[Pose, list[RefinementStep]]:
    """Levenberg-Marquardt on reprojection error (motion-only bundle adjustment).

    Parameters are a small update delta = (w, dt) applied to the current pose:

        R <- exp([w]x) R,    t <- exp([w]x) t + dt

    Each iteration:

    1. build residuals r(delta) = projected - observed pixels, as Values
    2. J = dr/d(delta) at delta = 0, one ``gradients`` call per residual
    3. solve (J^T J + lambda * diag(J^T J)) step = -J^T r
    4. keep the step if the error drops (and trust the model more: lower
       lambda), otherwise reject it and raise lambda
    """

    history = [RefinementStep(0, reprojection_rmse(camera, pose, points, pixels), damping)]
    for iteration in range(1, iterations + 1):
        delta = [Value(0.0) for _ in range(6)]
        residuals = _reprojection_residuals(camera, pose, points, pixels, delta)
        jacobian = [gradients(residual, delta) for residual in residuals]
        values = [residual.data for residual in residuals]

        jt_j = [
            [sum(row[i] * row[j] for row in jacobian) for j in range(6)]
            for i in range(6)
        ]
        jt_r = [sum(row[i] * r for row, r in zip(jacobian, values)) for i in range(6)]

        current = history[-1].rmse
        while True:
            damped = [
                [jt_j[i][j] + (damping * jt_j[i][i] if i == j else 0.0) for j in range(6)]
                for i in range(6)
            ]
            step = solve(damped, [-value for value in jt_r])
            candidate = _apply_update(pose, step)
            error = reprojection_rmse(camera, candidate, points, pixels)
            if error < current:
                pose = candidate
                damping = max(damping / 10.0, 1e-12)
                break
            damping *= 10.0
            if damping > 1e12:
                break
        history.append(RefinementStep(iteration, reprojection_rmse(camera, pose, points, pixels), damping))
        if current - history[-1].rmse < 1e-12:
            break
    return pose, history


def _reprojection_residuals(
    camera: PinholeCamera,
    pose: Pose,
    points: Sequence[Vector],
    pixels: Sequence[Vector],
    delta: list[Value],
) -> list[Value]:
    update = rotation_from_axis_angle(delta[:3])
    rotation = _mat_mul_generic(update, pose.rotation)
    translation = [
        sum(update[row][k] * pose.translation[k] for k in range(3)) + delta[3 + row]
        for row in range(3)
    ]

    residuals = []
    for point, (u, v) in zip(points, pixels):
        x, y, z = [
            sum(rotation[row][k] * point[k] for k in range(3)) + translation[row]
            for row in range(3)
        ]
        residuals.append(camera.fx * x / z + camera.cx - u)
        residuals.append(camera.fy * y / z + camera.cy - v)
    return residuals


def _apply_update(pose: Pose, step: Vector) -> Pose:
    update = rotation_from_axis_angle(step[:3])
    rotation = mat_mul(update, pose.rotation)
    translation = [
        value + offset
        for value, offset in zip(mat_vec(update, pose.translation), step[3:])
    ]
    return Pose(rotation, translation)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _projection_matrix(pose: Pose) -> Matrix:
    """[R | t] as a 3x4 matrix (normalized camera, so K = I)."""

    return [rotation_row + [t] for rotation_row, t in zip(pose.rotation, pose.translation)]


def _mat_mul_generic(left: list[list], right: list[list]) -> list[list]:
    """3x3 product that works for floats and autograd Values alike."""

    return [
        [sum(left[row][k] * right[k][col] for k in range(3)) for col in range(3)]
        for row in range(3)
    ]


def _sin(value):
    return value.sin() if isinstance(value, Value) else math.sin(value)


def _cos(value):
    return value.cos() if isinstance(value, Value) else math.cos(value)


def _sqrt(value):
    return value.sqrt() if isinstance(value, Value) else math.sqrt(value)

