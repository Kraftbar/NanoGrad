"""Bundle adjustment: refine camera poses and 3D points together.

After triangulation and PnP, poses and points each came from estimates that
ignored the others' errors. Bundle adjustment minimizes the pixel
reprojection error of *every* observation over *all* unknowns at once:

    unknowns:  every camera pose except the first (6 numbers each)
               + every 3D point (3 numbers each)
    residuals: projected - observed pixel, one 2D residual per observation
               (an observation = "point i was seen at this pixel in view v")

The first camera stays fixed at the world origin. Images cannot see the
scene's scale, so after each step the scene is rescaled to keep the first
baseline (camera 1's distance from camera 0) fixed; that changes nothing in
the images (it is a "gauge" freedom).

The Schur complement
--------------------
Each residual touches one camera and one point, so the Gauss-Newton matrix
has an arrow shape:

    H = [ A    B  ]      A: (6 V) x (6 V) camera block
        [ B^T  C  ]      C: block diagonal, one 3x3 block per point

Eliminating the points gives a small system for the cameras alone,

    (A - B C^-1 B^T) dp = -g_p + B C^-1 g_x

after which every point update is an independent 3x3 solve:

    dx_i = C_i^-1 (-g_i - B_i^T dp)

Point i couples the cameras that saw it: its term B_i C_i^-1 B_i^T fills the
(v, w) blocks for every pair of views v, w that observed it. The work grows
with the number of cameras, not the number of points, which is how real
bundle adjusters (g2o, Ceres, ORB-SLAM3) stay fast.
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
class Observation:
    """Point ``point`` was seen at ``pixel`` in view ``view``."""

    point: int
    view: int
    pixel: Vector


@dataclass(frozen=True)
class BundleResult:
    poses: list[Pose]                 # poses[0] is the fixed world origin
    points: list[Vector]              # same indexing as the input points
    observations: list[Observation]   # the observations that survived
    history: list[RefinementStep]

    def active_points(self) -> list[int]:
        """Points still seen in at least two views (the ones that were adjusted)."""

        return _active_points(self.observations)


@dataclass(frozen=True)
class TwoViewResult:
    pose: Pose
    points: list[Vector]        # only the kept points
    history: list[RefinementStep]
    kept: list[int]             # indices (into the inputs) of the kept points


@dataclass(frozen=True)
class _Linearized:
    """One observation's 2 residuals and their Jacobians at zero update."""

    point: int
    view: int
    pose_jacobian: Matrix | None   # 2 x 6, None for the fixed first view
    point_jacobian: Matrix         # 2 x 3
    residuals: Vector              # 2


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def bundle_adjust(
    camera: PinholeCamera,
    poses: Sequence[Pose],
    points: Sequence[Vector],
    observations: Sequence[Observation],
    *,
    iterations: int = 50,
    damping: float = 1e-3,
    huber_px: float | None = None,
    reject_px: float | None = None,
    rounds: int = 3,
) -> BundleResult:
    """Levenberg-Marquardt over all poses but the first and all points.

    With ``huber_px`` set, each observation's squared pixel error e^2 is
    replaced by the Huber cost

        rho(e) = e^2                    if e <= delta
                 2 delta e - delta^2    otherwise

    which grows only linearly for large errors, so a few wrong matches
    cannot dominate the fit. It is solved by iteratively reweighted least
    squares: each observation gets weight w = min(1, delta / e) and its
    Jacobian rows and residuals are scaled by sqrt(w). (ORB-SLAM3 uses the
    same Huber kernel in its g2o bundle adjustment.)

    With ``reject_px`` set, this runs up to ``rounds`` fits: after each fit,
    each point drops its *worst* observation if that one exceeds
    ``reject_px``, and the rest are adjusted again. Only the worst one: a
    single bad observation drags its point away, so the point's good
    observations can look bad too until the bad one is gone. A point left with fewer than two
    observations is no longer constrained and stops being adjusted. Use it
    together with ``huber_px``: under plain least squares a few bad matches
    can bend the fit so much that good observations get rejected.
    """

    if len(poses) < 2:
        raise ValueError("bundle adjustment needs at least two views")
    observations = list(observations)
    if not _active_points(observations):
        raise ValueError("bundle adjustment needs a point seen in two views")

    poses = list(poses)
    points = [list(point) for point in points]
    history: list[RefinementStep] = []

    def adjust() -> None:
        nonlocal poses, points
        poses, points, round_history = _adjust(
            camera,
            poses,
            points,
            observations,
            iterations=iterations,
            damping=damping,
            huber_px=huber_px,
        )
        # Number iterations continuously across rounds.
        offset = history[-1].iteration if history else 0
        steps = round_history if not history else round_history[1:]
        history.extend(
            RefinementStep(step.iteration + offset, step.rmse, step.damping) for step in steps
        )

    adjust()
    for _ in range((rounds - 1) if reject_px is not None else 0):
        survivors = _drop_worst_per_point(camera, poses, points, observations, reject_px)
        survivors = [
            observation
            for observation in survivors
            if observation.point in set(_active_points(survivors))
        ]
        if len(survivors) == len(observations) or len(_active_points(survivors)) < 6:
            break
        observations = survivors
        adjust()  # every prune is followed by a fit on the survivors
    return BundleResult(poses, points, observations, history)


def two_view_bundle_adjustment(
    camera: PinholeCamera,
    pose: Pose,
    points: Sequence[Vector],
    first_pixels: Sequence[Vector],
    second_pixels: Sequence[Vector],
    *,
    iterations: int = 50,
    damping: float = 1e-3,
    huber_px: float | None = None,
    reject_px: float | None = None,
    rounds: int = 3,
) -> TwoViewResult:
    """``bundle_adjust`` for two views: first camera at the origin, one pose.

    A point counts as kept only while both of its observations survive.
    """

    if not points:
        raise ValueError("bundle adjustment needs at least one point")

    observations = [
        Observation(index, view, pixel)
        for index, (first, second) in enumerate(zip(first_pixels, second_pixels))
        for view, pixel in ((0, first), (1, second))
    ]
    result = bundle_adjust(
        camera,
        [Pose.identity(), pose],
        points,
        observations,
        iterations=iterations,
        damping=damping,
        huber_px=huber_px,
        reject_px=reject_px,
        rounds=rounds,
    )
    kept = result.active_points()
    return TwoViewResult(
        result.poses[1],
        [result.points[index] for index in kept],
        result.history,
        kept,
    )


def reprojection_rmse(
    camera: PinholeCamera,
    poses: Sequence[Pose],
    points: Sequence[Vector],
    observations: Sequence[Observation],
) -> float:
    """Pixel RMSE over every observation."""

    total = 0.0
    for observation in observations:
        error = _pixel_error(camera, poses, points, observation)
        if math.isinf(error):
            return math.inf
        total += error * error
    return math.sqrt(total / len(observations))


def two_view_rmse(
    camera: PinholeCamera,
    pose: Pose,
    points: Sequence[Vector],
    first_pixels: Sequence[Vector],
    second_pixels: Sequence[Vector],
) -> float:
    """Pixel RMSE over both images (first camera at the origin)."""

    observations = [
        Observation(index, view, pixel)
        for index, (first, second) in enumerate(zip(first_pixels, second_pixels))
        for view, pixel in ((0, first), (1, second))
    ]
    return reprojection_rmse(camera, [Pose.identity(), pose], points, observations)


# ---------------------------------------------------------------------------
# One round of Levenberg-Marquardt
# ---------------------------------------------------------------------------


def _adjust(
    camera: PinholeCamera,
    poses: list[Pose],
    points: list[Vector],
    observations: list[Observation],
    *,
    iterations: int,
    damping: float,
    huber_px: float | None = None,
) -> tuple[list[Pose], list[Vector], list[RefinementStep]]:
    """Damped Gauss-Newton steps on a fixed set of observations."""

    active = _active_points(observations)
    used = [observation for observation in observations if observation.point in set(active)]
    baseline = norm(poses[1].translation)

    def rmse(candidate_poses: list[Pose], candidate_points: list[Vector]) -> float:
        return reprojection_rmse(camera, candidate_poses, candidate_points, used)

    def cost(candidate_poses: list[Pose], candidate_points: list[Vector]) -> float:
        if huber_px is None:
            return rmse(candidate_poses, candidate_points)
        return _huber_cost(camera, candidate_poses, candidate_points, used, huber_px)

    history = [RefinementStep(0, rmse(poses, points), damping)]
    current = cost(poses, points)
    for iteration in range(1, iterations + 1):
        rows = [_linearize(camera, poses, points, observation) for observation in used]
        if huber_px is not None:
            rows = [_huber_weighted(row, huber_px) for row in rows]
        previous = current

        while damping <= 1e12:
            pose_steps, point_steps = _schur_step(rows, active, len(poses), damping)
            candidate_poses = [poses[0]] + [
                apply_pose_update(pose, step) for pose, step in zip(poses[1:], pose_steps)
            ]
            candidate_points = [list(point) for point in points]
            for index, step in point_steps.items():
                candidate_points[index] = [value + offset for value, offset in zip(points[index], step)]
            candidate_poses, candidate_points = _fix_scale(candidate_poses, candidate_points, baseline)
            candidate_cost = cost(candidate_poses, candidate_points)
            if candidate_cost < current:
                poses, points, current = candidate_poses, candidate_points, candidate_cost
                damping = max(damping / 10.0, 1e-12)
                break
            damping *= 10.0

        history.append(RefinementStep(iteration, rmse(poses, points), damping))
        if previous - current < 1e-12:
            break
    return poses, points, history


def _linearize(
    camera: PinholeCamera,
    poses: list[Pose],
    points: list[Vector],
    observation: Observation,
) -> _Linearized:
    """Residuals and Jacobians at zero update, read off with autograd.

    The camera update is the same as in ``refine_pose``:
    R <- exp([w]x) R, t <- exp([w]x) t + dt.
    """

    pose = poses[observation.view]
    point_delta = [Value(0.0) for _ in range(3)]
    moved = [value + offset for value, offset in zip(points[observation.point], point_delta)]

    if observation.view == 0:
        pose_delta: list[Value] = []
        in_camera = [
            sum(pose.rotation[row][k] * moved[k] for k in range(3)) + pose.translation[row]
            for row in range(3)
        ]
    else:
        pose_delta = [Value(0.0) for _ in range(6)]
        update = rotation_from_axis_angle(pose_delta[:3])
        rotation = [
            [sum(update[row][k] * pose.rotation[k][col] for k in range(3)) for col in range(3)]
            for row in range(3)
        ]
        translation = [
            sum(update[row][k] * pose.translation[k] for k in range(3)) + pose_delta[3 + row]
            for row in range(3)
        ]
        in_camera = [
            sum(rotation[row][k] * moved[k] for k in range(3)) + translation[row]
            for row in range(3)
        ]

    residuals = _pixel_residuals(camera, in_camera, observation.pixel)
    inputs = [*pose_delta, *point_delta]
    jacobian = [gradients(residual, inputs) for residual in residuals]
    return _Linearized(
        point=observation.point,
        view=observation.view,
        pose_jacobian=[row[:6] for row in jacobian] if pose_delta else None,
        point_jacobian=[row[-3:] for row in jacobian],
        residuals=[residual.data for residual in residuals],
    )


def _pixel_residuals(camera: PinholeCamera, point_camera: list, observed: Vector) -> list:
    x, y, z = point_camera
    return [
        camera.fx * x / z + camera.cx - observed[0],
        camera.fy * y / z + camera.cy - observed[1],
    ]


def _schur_step(
    rows: list[_Linearized],
    active: list[int],
    view_count: int,
    damping: float,
) -> tuple[list[Vector], dict[int, Vector]]:
    """Solve the damped normal equations by eliminating the points first.

    Camera v (v >= 1) owns parameters 6 (v - 1) .. 6 (v - 1) + 5.
    """

    size = 6 * (view_count - 1)
    reduced = [[0.0] * size for _ in range(size)]   # A - sum B_i C_i^-1 B_i^T
    reduced_rhs = [0.0] * size                       # -g_p + sum B_i C_i^-1 g_i
    pose_diagonal = [0.0] * size                     # diag(A), for damping

    by_point: dict[int, list[_Linearized]] = {index: [] for index in active}
    for row in rows:
        by_point[row.point].append(row)

    eliminated = {}
    for index, point_rows in by_point.items():
        c = [[0.0] * 3 for _ in range(3)]
        g_x = [0.0] * 3
        couplings: dict[int, Matrix] = {}     # view -> B_iv (6x3)
        for row in point_rows:
            jx_t = transpose(row.point_jacobian)
            _add_into(c, mat_mul(jx_t, row.point_jacobian))
            g_x = [a + b for a, b in zip(g_x, mat_vec(jx_t, row.residuals))]
            if row.pose_jacobian is None:
                continue
            jp_t = transpose(row.pose_jacobian)
            start = 6 * (row.view - 1)
            a = mat_mul(jp_t, row.pose_jacobian)
            g_p = mat_vec(jp_t, row.residuals)
            for r in range(6):
                reduced_rhs[start + r] -= g_p[r]
                pose_diagonal[start + r] += a[r][r]
                for col in range(6):
                    reduced[start + r][start + col] += a[r][col]
            coupling = couplings.setdefault(row.view, [[0.0] * 3 for _ in range(6)])
            _add_into(coupling, mat_mul(jp_t, row.point_jacobian))

        c_inverse = inverse(_damped(c, damping))
        for view, coupling in couplings.items():
            b_c_inverse = mat_mul(coupling, c_inverse)          # 6x3
            b_c_g = mat_vec(b_c_inverse, g_x)
            start = 6 * (view - 1)
            for r in range(6):
                reduced_rhs[start + r] += b_c_g[r]
            for other_view, other in couplings.items():
                block = mat_mul(b_c_inverse, transpose(other))  # 6x6, cameras (view, other_view)
                other_start = 6 * (other_view - 1)
                for r in range(6):
                    for col in range(6):
                        reduced[start + r][other_start + col] -= block[r][col]
        eliminated[index] = (couplings, c_inverse, g_x)

    # Damping on the camera block: added after the sum so it scales with A.
    for i in range(size):
        reduced[i][i] += damping * pose_diagonal[i]
    pose_step = solve(reduced, reduced_rhs)
    pose_steps = [pose_step[6 * v : 6 * v + 6] for v in range(view_count - 1)]

    point_steps = {}
    for index, (couplings, c_inverse, g_x) in eliminated.items():
        rhs = [-value for value in g_x]
        for view, coupling in couplings.items():
            b_t_dp = mat_vec(transpose(coupling), pose_steps[view - 1])
            rhs = [value - v for value, v in zip(rhs, b_t_dp)]
        point_steps[index] = mat_vec(c_inverse, rhs)
    return pose_steps, point_steps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _drop_worst_per_point(
    camera: PinholeCamera,
    poses: Sequence[Pose],
    points: Sequence[Vector],
    observations: Sequence[Observation],
    reject_px: float,
) -> list[Observation]:
    worst: dict[int, tuple[float, int]] = {}
    for position, observation in enumerate(observations):
        error = _pixel_error(camera, poses, points, observation)
        if error > reject_px and error > worst.get(observation.point, (-1.0, -1))[0]:
            worst[observation.point] = (error, position)
    dropped = {position for _, position in worst.values()}
    return [observation for position, observation in enumerate(observations) if position not in dropped]


def _active_points(observations: Sequence[Observation]) -> list[int]:
    counts: dict[int, int] = {}
    for observation in observations:
        counts[observation.point] = counts.get(observation.point, 0) + 1
    return sorted(index for index, count in counts.items() if count >= 2)


def _pixel_error(
    camera: PinholeCamera,
    poses: Sequence[Pose],
    points: Sequence[Vector],
    observation: Observation,
) -> float:
    projected = camera.project(poses[observation.view].apply(points[observation.point]))
    if projected is None:
        return math.inf
    return math.dist(projected, observation.pixel)


def _huber_weighted(row: _Linearized, delta: float) -> _Linearized:
    """Scale an observation's rows by sqrt(w), w = min(1, delta / error)."""

    error = math.hypot(*row.residuals)
    scale = 1.0 if error <= delta else math.sqrt(delta / error)

    def scaled(matrix: Matrix | None) -> Matrix | None:
        if matrix is None:
            return None
        return [[scale * value for value in values] for values in matrix]

    return _Linearized(
        row.point,
        row.view,
        scaled(row.pose_jacobian),
        scaled(row.point_jacobian),
        [scale * value for value in row.residuals],
    )


def _huber_cost(
    camera: PinholeCamera,
    poses: Sequence[Pose],
    points: Sequence[Vector],
    observations: Sequence[Observation],
    delta: float,
) -> float:
    total = 0.0
    for observation in observations:
        error = _pixel_error(camera, poses, points, observation)
        if math.isinf(error):
            return math.inf
        total += error * error if error <= delta else 2.0 * delta * error - delta * delta
    return total


def _add_into(target: Matrix, values: Matrix) -> None:
    for target_row, value_row in zip(target, values):
        for col, value in enumerate(value_row):
            target_row[col] += value


def _damped(matrix: Matrix, damping: float) -> Matrix:
    return [
        [value + (damping * value if row == col else 0.0) for col, value in enumerate(values)]
        for row, values in enumerate(matrix)
    ]


def _fix_scale(
    poses: list[Pose],
    points: list[Vector],
    baseline: float,
) -> tuple[list[Pose], list[Vector]]:
    """Rescale the scene so camera 1 keeps its distance from camera 0.

    Scaling every point and every translation by s leaves all pixels
    unchanged: R (s X) + s t = s (R X + t) projects to the same place.
    """

    length = norm(poses[1].translation)
    if length == 0.0 or baseline == 0.0:
        # Pure rotation: there is no baseline to hold fixed.
        return poses, points
    scale = baseline / length
    return (
        [Pose(pose.rotation, [value * scale for value in pose.translation]) for pose in poses],
        [[value * scale for value in point] for point in points],
    )
