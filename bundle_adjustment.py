"""Two-view bundle adjustment: refine a camera pose and the 3D points together.

After the essential matrix and triangulation, the pose and the points each
came from a linear estimate that ignored the other's errors. Bundle
adjustment minimizes the pixel reprojection error in *both* images over
*all* unknowns at once:

    unknowns:  second camera pose (6 numbers) + every 3D point (3 each)
    residuals: projected - observed pixel, for each point in each image

The first camera stays fixed at the world origin. Two views cannot see
scale, so after each step the scene is rescaled to keep the baseline
length fixed; that changes nothing in the images (it is a "gauge" freedom).

The Schur complement
--------------------
With N points there are 6 + 3N unknowns, but each residual touches only the
pose and one point. The Gauss-Newton matrix therefore has an arrow shape:

    H = [ A    B  ]      A: 6x6 pose block
        [ B^T  C  ]      C: block diagonal, one 3x3 block per point

Eliminating the points gives a 6x6 system for the pose alone,

    (A - B C^-1 B^T) dp = -g_p + B C^-1 g_x

after which every point update is an independent 3x3 solve:

    dx_i = C_i^-1 (-g_i - B_i^T dp)

That turns one (6 + 3N)-sized solve into a 6x6 solve plus N 3x3 solves,
which is how real bundle adjusters (g2o, Ceres, ORB-SLAM3) stay fast.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from engine import Value, gradients
from geometry import (
    PinholeCamera,
    Pose,
    RefinementStep,
    apply_pose_update,
    rotation_from_axis_angle,
)
from linalg import Matrix, Vector, inverse, mat_mul, mat_vec, norm, solve, transpose


@dataclass(frozen=True)
class TwoViewResult:
    pose: Pose
    points: list[Vector]
    history: list[RefinementStep]


@dataclass(frozen=True)
class _PointBlock:
    """Linearization of one point's 4 residuals (u, v in each image)."""

    pose_jacobian: Matrix   # 4 x 6
    point_jacobian: Matrix  # 4 x 3
    residuals: Vector       # 4


def two_view_bundle_adjustment(
    camera: PinholeCamera,
    pose: Pose,
    points: Sequence[Vector],
    first_pixels: Sequence[Vector],
    second_pixels: Sequence[Vector],
    *,
    iterations: int = 10,
    damping: float = 1e-3,
) -> TwoViewResult:
    """Levenberg-Marquardt over (second pose, all points) via the Schur complement."""

    points = [list(point) for point in points]
    baseline = norm(pose.translation)

    def error(candidate_pose: Pose, candidate_points: list[Vector]) -> float:
        return two_view_rmse(camera, candidate_pose, candidate_points, first_pixels, second_pixels)

    history = [RefinementStep(0, error(pose, points), damping)]
    for iteration in range(1, iterations + 1):
        blocks = [
            _linearize_point(camera, pose, point, first, second)
            for point, first, second in zip(points, first_pixels, second_pixels)
        ]
        current = history[-1].rmse

        while damping <= 1e12:
            pose_step, point_steps = _schur_step(blocks, damping)
            candidate_pose = apply_pose_update(pose, pose_step)
            candidate_points = [
                [value + offset for value, offset in zip(point, step)]
                for point, step in zip(points, point_steps)
            ]
            candidate_pose, candidate_points = _fix_scale(candidate_pose, candidate_points, baseline)
            if error(candidate_pose, candidate_points) < current:
                pose, points = candidate_pose, candidate_points
                damping = max(damping / 10.0, 1e-12)
                break
            damping *= 10.0

        history.append(RefinementStep(iteration, error(pose, points), damping))
        if current - history[-1].rmse < 1e-12:
            break
    return TwoViewResult(pose, points, history)


def two_view_rmse(
    camera: PinholeCamera,
    pose: Pose,
    points: Sequence[Vector],
    first_pixels: Sequence[Vector],
    second_pixels: Sequence[Vector],
) -> float:
    """Pixel RMSE over both images (first camera at the origin)."""

    first_pose = Pose.identity()
    total = 0.0
    count = 0
    for point, first, second in zip(points, first_pixels, second_pixels):
        for view_pose, observed in ((first_pose, first), (pose, second)):
            projected = camera.project(view_pose.apply(point))
            if projected is None:
                return math.inf
            total += (projected[0] - observed[0]) ** 2 + (projected[1] - observed[1]) ** 2
            count += 1
    return math.sqrt(total / count)


def _linearize_point(
    camera: PinholeCamera,
    pose: Pose,
    point: Vector,
    first_pixel: Vector,
    second_pixel: Vector,
) -> _PointBlock:
    """Residuals and Jacobians at zero update, read off with autograd."""

    pose_delta = [Value(0.0) for _ in range(6)]
    point_delta = [Value(0.0) for _ in range(3)]
    moved = [value + offset for value, offset in zip(point, point_delta)]

    update = rotation_from_axis_angle(pose_delta[:3])
    rotation = [
        [sum(update[row][k] * pose.rotation[k][col] for k in range(3)) for col in range(3)]
        for row in range(3)
    ]
    translation = [
        sum(update[row][k] * pose.translation[k] for k in range(3)) + pose_delta[3 + row]
        for row in range(3)
    ]
    in_second = [
        sum(rotation[row][k] * moved[k] for k in range(3)) + translation[row]
        for row in range(3)
    ]

    residuals = [
        *_pixel_residuals(camera, moved, first_pixel),
        *_pixel_residuals(camera, in_second, second_pixel),
    ]
    inputs = [*pose_delta, *point_delta]
    jacobian = [gradients(residual, inputs) for residual in residuals]
    return _PointBlock(
        pose_jacobian=[row[:6] for row in jacobian],
        point_jacobian=[row[6:] for row in jacobian],
        residuals=[residual.data for residual in residuals],
    )


def _pixel_residuals(camera: PinholeCamera, point_camera: list, observed: Vector) -> list:
    x, y, z = point_camera
    return [
        camera.fx * x / z + camera.cx - observed[0],
        camera.fy * y / z + camera.cy - observed[1],
    ]


def _schur_step(blocks: list[_PointBlock], damping: float) -> tuple[Vector, list[Vector]]:
    """Solve the damped normal equations by eliminating the points first."""

    reduced = [[0.0] * 6 for _ in range(6)]    # A - sum B_i C_i^-1 B_i^T
    reduced_rhs = [0.0] * 6                     # -g_p + sum B_i C_i^-1 g_i
    eliminated = []
    for block in blocks:
        jp_t = transpose(block.pose_jacobian)
        jx_t = transpose(block.point_jacobian)
        a = mat_mul(jp_t, block.pose_jacobian)          # 6x6
        b = mat_mul(jp_t, block.point_jacobian)         # 6x3
        c = _damped(mat_mul(jx_t, block.point_jacobian), damping)  # 3x3
        g_p = mat_vec(jp_t, block.residuals)            # 6
        g_x = mat_vec(jx_t, block.residuals)            # 3

        c_inverse = inverse(c)
        b_c_inverse = mat_mul(b, c_inverse)             # 6x3
        b_c_b = mat_mul(b_c_inverse, transpose(b))      # 6x6
        b_c_g = mat_vec(b_c_inverse, g_x)               # 6
        for row in range(6):
            reduced_rhs[row] += -g_p[row] + b_c_g[row]
            for col in range(6):
                reduced[row][col] += a[row][col] - b_c_b[row][col]
        eliminated.append((b, c_inverse, g_x))

    # Damping on the pose block: added after the sum so it scales with A.
    pose_diagonal = [
        sum(sum(row[i] * row[i] for row in block.pose_jacobian) for block in blocks)
        for i in range(6)
    ]
    for i in range(6):
        reduced[i][i] += damping * pose_diagonal[i]
    pose_step = solve(reduced, reduced_rhs)

    point_steps = []
    for b, c_inverse, g_x in eliminated:
        b_t_dp = mat_vec(transpose(b), pose_step)
        point_steps.append(mat_vec(c_inverse, [-g - v for g, v in zip(g_x, b_t_dp)]))
    return pose_step, point_steps


def _damped(matrix: Matrix, damping: float) -> Matrix:
    return [
        [value + (damping * value if row == col else 0.0) for col, value in enumerate(values)]
        for row, values in enumerate(matrix)
    ]


def _fix_scale(pose: Pose, points: list[Vector], baseline: float) -> tuple[Pose, list[Vector]]:
    """Rescale the scene so the baseline keeps its length (no change in pixels)."""

    length = norm(pose.translation)
    if length == 0.0:
        return pose, points
    scale = baseline / length
    return (
        Pose(pose.rotation, [value * scale for value in pose.translation]),
        [[value * scale for value in point] for point in points],
    )
