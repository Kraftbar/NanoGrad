"""Camera pose estimation on a synthetic scene, no downloads needed.

Two classic problems, each checked against ground truth:

1. Absolute pose (PnP): known 3D points + their pixels in a new camera ->
   where is that camera? RANSAC around linear DLT throws out wrong matches,
   then Levenberg-Marquardt refines the pose, with Jacobians from NanoGrad's
   scalar autograd.
2. Relative pose (two-view): pixels matched between two cameras -> how did
   the camera move? RANSAC around the eight-point essential matrix, a
   cheirality check, triangulation, then two-view bundle adjustment of the
   pose and every point together (up to scale).

``--outliers 0.3`` replaces 30% of the second image's matches with random
pixels, to show what RANSAC is for.

These are the geometric building blocks underneath feature-based SLAM such
as ORB-SLAM3 (PnP for tracking, essential matrix for initialization, bundle
adjustment for refinement). See docs/camera_geometry.md.

    python3 pose_demo.py
    python3 pose_demo.py --noise 2.0 --points 60 --seed 3
    python3 pose_demo.py --outliers 0.3
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass

from bundle_adjustment import two_view_bundle_adjustment
from geometry import (
    PinholeCamera,
    Pose,
    essential_matrix,
    estimate_pose_dlt,
    refine_pose,
    relative_pose_from_essential,
    reprojection_rmse,
    rotation_angle_degrees,
    triangulate_point,
)
from linalg import Vector, dot, norm, normalize
from ransac import ransac_pose_dlt, ransac_relative_pose


DEFAULT_CAMERA = PinholeCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)


@dataclass(frozen=True)
class Scene:
    camera: PinholeCamera
    points: list[Vector]
    first_pose: Pose
    second_pose: Pose
    first_pixels: list[Vector]
    second_pixels: list[Vector]
    outliers: list[int]


def synthetic_scene(
    *,
    points: int = 40,
    noise: float = 0.5,
    outliers: float = 0.0,
    seed: int = 0,
    camera: PinholeCamera = DEFAULT_CAMERA,
) -> Scene:
    """Random points in a box seen by two cameras, with Gaussian pixel noise.

    The first camera sits at the world origin. The second has moved right and
    forward and turned a little toward the points, like a handheld
    camera stepping sideways.
    """

    rng = random.Random(seed)
    first_pose = Pose.identity()
    second_pose = Pose.from_axis_angle([0.05, 0.15, 0.03], [-0.8, 0.05, 0.1])

    world_points: list[Vector] = []
    first_pixels: list[Vector] = []
    second_pixels: list[Vector] = []
    while len(world_points) < points:
        point = [rng.uniform(-2.0, 2.0), rng.uniform(-1.5, 1.5), rng.uniform(4.0, 8.0)]
        first = camera.project(first_pose.apply(point))
        second = camera.project(second_pose.apply(point))
        if not (camera.contains(first) and camera.contains(second)):
            continue
        world_points.append(point)
        first_pixels.append([value + rng.gauss(0.0, noise) for value in first])
        second_pixels.append([value + rng.gauss(0.0, noise) for value in second])

    # Wrong matches: the second-image pixel is replaced by a random one.
    outlier_indices = sorted(rng.sample(range(points), round(outliers * points)))
    for index in outlier_indices:
        second_pixels[index] = [rng.uniform(0.0, camera.width), rng.uniform(0.0, camera.height)]

    return Scene(
        camera,
        world_points,
        first_pose,
        second_pose,
        first_pixels,
        second_pixels,
        outlier_indices,
    )


def translation_error(estimate: Pose, truth: Pose) -> float:
    return norm([a - b for a, b in zip(estimate.translation, truth.translation)])


def direction_error_degrees(estimate: Vector, truth: Vector) -> float:
    cos_angle = dot(normalize(estimate), normalize(truth))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))


def absolute_pose_report(
    scene: Scene,
    *,
    iterations: int = 10,
    threshold_px: float = 2.0,
) -> dict:
    """PnP for the second camera: plain DLT, RANSAC DLT, then LM on inliers."""

    camera = scene.camera
    normalized = [camera.normalize(pixel) for pixel in scene.second_pixels]
    plain = estimate_pose_dlt(scene.points, normalized)
    robust = ransac_pose_dlt(scene.points, normalized, threshold=threshold_px / camera.fx)
    inliers = robust.inliers
    inlier_points = [scene.points[i] for i in inliers]
    inlier_pixels = [scene.second_pixels[i] for i in inliers]

    refined, history = refine_pose(
        camera,
        robust.model,
        inlier_points,
        inlier_pixels,
        iterations=iterations,
    )
    truth = scene.second_pose
    return {
        "plain_rotation_error_deg": rotation_angle_degrees(plain.rotation, truth.rotation),
        "linear_rotation_error_deg": rotation_angle_degrees(robust.model.rotation, truth.rotation),
        "linear_translation_error": translation_error(robust.model, truth),
        "linear_rmse_px": reprojection_rmse(camera, robust.model, inlier_points, inlier_pixels),
        "refined_rotation_error_deg": rotation_angle_degrees(refined.rotation, truth.rotation),
        "refined_translation_error": translation_error(refined, truth),
        "refined_rmse_px": history[-1].rmse,
        "truth_rmse_px": reprojection_rmse(camera, truth, inlier_points, inlier_pixels),
        "inliers": inliers,
        "outlier_recall": _outlier_recall(scene, inliers),
        "ransac_iterations": robust.iterations,
        "iterations": len(history) - 1,
        "history": history,
    }


def relative_pose_report(scene: Scene, *, threshold_px: float = 2.0) -> dict:
    """Two-view geometry: RANSAC essential matrix, triangulation, bundle adjustment."""

    camera = scene.camera
    first = [camera.normalize(pixel) for pixel in scene.first_pixels]
    second = [camera.normalize(pixel) for pixel in scene.second_pixels]
    plain = relative_pose_from_essential(essential_matrix(first, second), first, second)
    relative, robust = ransac_relative_pose(first, second, threshold=threshold_px / camera.fx)
    inliers = robust.inliers
    truth = scene.second_pose  # the first camera is the world frame

    triangulated = [
        triangulate_point(Pose.identity(), relative, first[i], second[i])
        for i in inliers
    ]
    adjusted = two_view_bundle_adjustment(
        camera,
        relative,
        triangulated,
        [scene.first_pixels[i] for i in inliers],
        [scene.second_pixels[i] for i in inliers],
    )

    # Two views fix the scene only up to scale; align it using the true baseline.
    scale = norm(truth.translation)
    true_points = [scene.points[i] for i in inliers]

    def median_point_error(points: list[Vector]) -> float:
        errors = sorted(
            norm([scale * a - b for a, b in zip(estimate, point)])
            for estimate, point in zip(points, true_points)
        )
        return errors[len(errors) // 2]

    return {
        "plain_rotation_error_deg": rotation_angle_degrees(plain.rotation, truth.rotation),
        "rotation_error_deg": rotation_angle_degrees(relative.rotation, truth.rotation),
        "translation_direction_error_deg": direction_error_degrees(
            relative.translation,
            truth.translation,
        ),
        "median_point_error": median_point_error(triangulated),
        "ba_rotation_error_deg": rotation_angle_degrees(adjusted.pose.rotation, truth.rotation),
        "ba_translation_direction_error_deg": direction_error_degrees(
            adjusted.pose.translation,
            truth.translation,
        ),
        "ba_median_point_error": median_point_error(adjusted.points),
        "ba_history": adjusted.history,
        "inliers": inliers,
        "outlier_recall": _outlier_recall(scene, inliers),
        "mean_scene_depth": sum(point[2] for point in scene.points) / len(scene.points),
    }


def _outlier_recall(scene: Scene, inliers: list[int]) -> float | None:
    """Fraction of the planted wrong matches that RANSAC rejected."""

    if not scene.outliers:
        return None
    kept = set(inliers)
    return sum(index not in kept for index in scene.outliers) / len(scene.outliers)


def ascii_view(camera: PinholeCamera, pixels: list[Vector], *, columns: int = 64, rows: int = 16) -> str:
    """Tiny text rendering of where points land in the image."""

    grid = [[" "] * columns for _ in range(rows)]
    for u, v in pixels:
        col = min(columns - 1, max(0, int(u / camera.width * columns)))
        row = min(rows - 1, max(0, int(v / camera.height * rows)))
        grid[row][col] = "*"
    border = "+" + "-" * columns + "+"
    return "\n".join([border, *("|" + "".join(line) + "|" for line in grid), border])


def run(args: argparse.Namespace) -> dict:
    scene = synthetic_scene(
        points=args.points,
        noise=args.noise,
        outliers=args.outliers,
        seed=args.seed,
    )
    absolute = absolute_pose_report(scene, iterations=args.iterations, threshold_px=args.threshold)
    relative = relative_pose_report(scene, threshold_px=args.threshold)

    print("NanoGrad camera pose demo")
    print(f"points: {args.points}, pixel noise: {args.noise} px, "
          f"wrong matches: {len(scene.outliers)}, seed: {args.seed}")
    print()
    print("second camera view:")
    print(ascii_view(scene.camera, scene.second_pixels))
    print()
    print("1) absolute pose (PnP) of the second camera")
    print(f"   DLT on all matches:  rotation err {absolute['plain_rotation_error_deg']:.3f} deg")
    print(f"   RANSAC DLT:          rotation err {absolute['linear_rotation_error_deg']:.3f} deg, "
          f"translation err {absolute['linear_translation_error']:.4f}, "
          f"{len(absolute['inliers'])} inliers{_recall_text(absolute)}")
    for step in absolute["history"]:
        print(f"   LM iter {step.iteration:>2}:  reprojection {step.rmse:.4f} px  (lambda {step.damping:.0e})")
    print(f"   refined:             rotation err {absolute['refined_rotation_error_deg']:.3f} deg, "
          f"translation err {absolute['refined_translation_error']:.4f}")
    print(f"   ground truth reprojection on the same pixels: {absolute['truth_rmse_px']:.4f} px")
    print()
    print("2) relative pose from matches only (essential matrix)")
    print(f"   8-point on all matches: rotation err {relative['plain_rotation_error_deg']:.3f} deg")
    print(f"   RANSAC 8-point:         rotation err {relative['rotation_error_deg']:.3f} deg, "
          f"translation direction err {relative['translation_direction_error_deg']:.3f} deg, "
          f"{len(relative['inliers'])} inliers{_recall_text(relative)}")
    for step in relative["ba_history"]:
        print(f"   BA iter {step.iteration:>2}:  reprojection {step.rmse:.4f} px (both images)")
    print(f"   bundle adjusted:        rotation err {relative['ba_rotation_error_deg']:.3f} deg, "
          f"translation direction err {relative['ba_translation_direction_error_deg']:.3f} deg")
    print(f"   triangulated points: median error {relative['median_point_error']:.4f} -> "
          f"{relative['ba_median_point_error']:.4f} after BA "
          f"(scene depth ~{relative['mean_scene_depth']:.1f})")
    return {"absolute": absolute, "relative": relative}


def _recall_text(report: dict) -> str:
    if report["outlier_recall"] is None:
        return ""
    return f" ({report['outlier_recall']:.0%} of wrong matches rejected)"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--points", type=int, default=40)
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outliers", type=float, default=0.0, help="fraction of wrong matches")
    parser.add_argument("--threshold", type=float, default=2.0, help="RANSAC inlier threshold in px")
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args(argv)
    if args.points < 8:
        parser.error("--points must be at least 8")
    if args.noise < 0.0:
        parser.error("--noise must be non-negative")
    if not 0.0 <= args.outliers < 0.8:
        parser.error("--outliers must be in [0, 0.8)")
    return args


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
