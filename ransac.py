"""RANSAC: fit models to data where some matches are simply wrong.

Least squares lets every point vote, so one bad match can drag the answer
anywhere. RANSAC (random sample consensus) instead repeats:

1. fit a model to a random *minimal* sample (6 points for PnP, 8 for E)
2. count the inliers: data within ``threshold`` of that model
3. keep the model with the most inliers, after refitting it on all of its
   inliers until that set stops growing (local optimization)

How many tries? If a fraction ``w`` of the data are inliers, one sample of
size ``s`` is all-inlier with probability ``w^s``. To succeed at least once
with confidence ``p`` we need

    iterations >= log(1 - p) / log(1 - w^s)

and ``w`` is re-estimated from the best inlier count as the search goes.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from geometry import (
    Pose,
    epipolar_matrix,
    essential_matrix,
    estimate_pose_dlt,
    relative_pose_from_essential,
)
from linalg import Matrix, Vector, mat_vec, transpose


Model = TypeVar("Model")


@dataclass(frozen=True)
class RansacResult(Generic[Model]):
    model: Model
    inliers: list[int]
    iterations: int


def ransac(
    count: int,
    sample_size: int,
    fit: Callable[[list[int]], Model],
    errors: Callable[[Model], list[float]],
    threshold: float,
    *,
    confidence: float = 0.999,
    max_iterations: int = 1000,
    rng: random.Random | None = None,
) -> RansacResult[Model]:
    """Generic RANSAC over ``count`` data items, referred to by index.

    ``fit(indices)`` builds a model from those items (it may raise
    ValueError for degenerate samples, which are skipped) and
    ``errors(model)`` scores every item.
    """

    if count < sample_size:
        raise ValueError("not enough data for one minimal sample")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    rng = rng or random.Random(0)

    def score(model: Model) -> list[int]:
        return [index for index, error in enumerate(errors(model)) if error < threshold]

    best_model: Model | None = None
    best_inliers: list[int] = []
    needed = max_iterations
    iteration = 0
    while iteration < min(needed, max_iterations):
        iteration += 1
        sample = rng.sample(range(count), sample_size)
        try:
            model = fit(sample)
        except ValueError:
            continue
        inliers = score(model)
        if len(inliers) > len(best_inliers) and len(inliers) >= sample_size:
            best_model, best_inliers = _grow(model, inliers, fit, score)
            needed = required_iterations(len(best_inliers) / count, sample_size, confidence)

    if best_model is None:
        raise ValueError("RANSAC found no model with enough inliers")
    return RansacResult(best_model, best_inliers, iteration)


def _grow(
    model: Model,
    inliers: list[int],
    fit: Callable[[list[int]], Model],
    score: Callable[[Model], list[int]],
) -> tuple[Model, list[int]]:
    """Local optimization: refit on all inliers until the set stops growing.

    A minimal sample is noisy, so the model it gives is a little off and
    misses some true inliers. Refitting on every inlier averages the noise
    out, which usually reveals more inliers, and so on. Doing this for every
    new best model (not just at the end) is the "LO-RANSAC" idea: it also
    makes the inlier ratio, and so the stopping rule, more accurate.
    """

    for _ in range(10):
        try:
            refined = fit(inliers)
        except ValueError:
            break
        refined_inliers = score(refined)
        if len(refined_inliers) <= len(inliers):
            break
        model, inliers = refined, refined_inliers
    return model, inliers


def required_iterations(inlier_ratio: float, sample_size: int, confidence: float) -> int:
    """Samples needed to draw one all-inlier sample with ``confidence``."""

    all_inlier = inlier_ratio ** sample_size
    if all_inlier >= 1.0:
        return 1
    if all_inlier <= 0.0:
        return 10**9
    return math.ceil(math.log(1.0 - confidence) / math.log(1.0 - all_inlier))


# ---------------------------------------------------------------------------
# Robust pose estimators
# ---------------------------------------------------------------------------


def ransac_pose_dlt(
    points: Sequence[Vector],
    normalized: Sequence[Vector],
    *,
    threshold: float,
    rng: random.Random | None = None,
) -> RansacResult[Pose]:
    """Robust PnP. ``threshold`` is in normalized units (pixels / focal)."""

    def fit(indices: list[int]) -> Pose:
        return estimate_pose_dlt(
            [points[i] for i in indices],
            [normalized[i] for i in indices],
        )

    def errors(pose: Pose) -> list[float]:
        return [
            _normalized_reprojection_error(pose, point, observed)
            for point, observed in zip(points, normalized)
        ]

    return ransac(len(points), 6, fit, errors, threshold, rng=rng)


def ransac_relative_pose(
    first_normalized: Sequence[Vector],
    second_normalized: Sequence[Vector],
    *,
    threshold: float,
    rng: random.Random | None = None,
) -> tuple[Pose, RansacResult[Matrix]]:
    """Robust two-view geometry: RANSAC on the epipolar constraint, then pose.

    Inliers are scored with the Sampson distance, a first-order estimate of
    how far (in normalized image units) a match is from satisfying
    x2^T E x1 = 0. It is cheap and needs no triangulation.

    Samples are scored with the rank-2 ``epipolar_matrix``, not its
    projection onto a true essential matrix. That projection can move the
    model enough (especially with a short baseline) to push real matches
    past the threshold. The essential constraint is enforced once, at the
    end, on all inliers.
    """

    def fit(indices: list[int]) -> Matrix:
        return epipolar_matrix(
            [first_normalized[i] for i in indices],
            [second_normalized[i] for i in indices],
        )

    def errors(essential: Matrix) -> list[float]:
        return [
            math.sqrt(sampson_distance_sq(essential, n1, n2))
            for n1, n2 in zip(first_normalized, second_normalized)
        ]

    result = ransac(len(first_normalized), 8, fit, errors, threshold, rng=rng)
    first_inliers = [first_normalized[i] for i in result.inliers]
    second_inliers = [second_normalized[i] for i in result.inliers]
    essential = essential_matrix(first_inliers, second_inliers)
    pose = relative_pose_from_essential(essential, first_inliers, second_inliers)
    return pose, result


def sampson_distance_sq(essential: Matrix, first: Vector, second: Vector) -> float:
    """Squared Sampson distance of one match to the epipolar constraint.

        e = x2^T E x1,   a = E x1,   b = E^T x2
        d^2 = e^2 / (a_0^2 + a_1^2 + b_0^2 + b_1^2)
    """

    x1 = [*first, 1.0]
    x2 = [*second, 1.0]
    a = mat_vec(essential, x1)
    b = mat_vec(transpose(essential), x2)
    e = sum(u * v for u, v in zip(x2, a))
    denominator = a[0] ** 2 + a[1] ** 2 + b[0] ** 2 + b[1] ** 2
    if denominator == 0.0:
        return math.inf
    return e * e / denominator


def _normalized_reprojection_error(pose: Pose, point: Vector, observed: Vector) -> float:
    x, y, z = pose.apply(point)
    if z <= 0.0:
        return math.inf
    return math.hypot(x / z - observed[0], y / z - observed[1])
