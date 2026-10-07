"""Camera pose estimation on a synthetic scene, no downloads needed.

Two classic problems, each checked against ground truth:

1. Absolute pose (PnP): known 3D points + their pixels in a new camera ->
   where is that camera? Linear DLT first, then Levenberg-Marquardt
   refinement with Jacobians from NanoGrad's scalar autograd.
2. Relative pose (two-view): pixels matched between two cameras -> how did
   the camera move? Eight-point essential matrix, cheirality check, then
   triangulation to rebuild the scene (up to scale).

These are the geometric building blocks underneath feature-based SLAM such
as ORB-SLAM3 (PnP for tracking, essential matrix for initialization, bundle
adjustment for refinement). See docs/camera_geometry.md.

    python3 pose_demo.py
    python3 pose_demo.py --noise 2.0 --points 60 --seed 3
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass

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


DEFAULT_CAMERA = PinholeCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)


@dataclass(frozen=True)
class Scene:
    camera: PinholeCamera
    points: list[Vector]
    first_pose: Pose
    second_pose: Pose
    first_pixels: list[Vector]
    second_pixels: list[Vector]


def synthetic_scene(
    *,
    points: int = 40,
    noise: float = 0.5,
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

    return Scene(camera, world_points, first_pose, second_pose, first_pixels, second_pixels)


def translation_error(estimate: Pose, truth: Pose) -> float:
    return norm([a - b for a, b in zip(estimate.translation, truth.translation)])


def direction_error_degrees(estimate: Vector, truth: Vector) -> float:
    cos_angle = dot(normalize(estimate), normalize(truth))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))


def absolute_pose_report(scene: Scene, *, iterations: int = 10) -> dict:
    """PnP for the second camera: DLT, then LM refinement."""

    camera = scene.camera
    normalized = [camera.normalize(pixel) for pixel in scene.second_pixels]
    linear = estimate_pose_dlt(scene.points, normalized)
    refined, history = refine_pose(
        camera,
        linear,
        scene.points,
        scene.second_pixels,
        iterations=iterations,
    )
    truth = scene.second_pose
    return {
        "linear_rotation_error_deg": rotation_angle_degrees(linear.rotation, truth.rotation),
        "linear_translation_error": translation_error(linear, truth),
        "linear_rmse_px": reprojection_rmse(camera, linear, scene.points, scene.second_pixels),
        "refined_rotation_error_deg": rotation_angle_degrees(refined.rotation, truth.rotation),
        "refined_translation_error": translation_error(refined, truth),
        "refined_rmse_px": history[-1].rmse,
        "truth_rmse_px": reprojection_rmse(camera, truth, scene.points, scene.second_pixels),
        "iterations": len(history) - 1,
        "history": history,
    }


def relative_pose_report(scene: Scene) -> dict:
    """Two-view geometry: essential matrix, relative pose, triangulation."""

    camera = scene.camera
    first = [camera.normalize(pixel) for pixel in scene.first_pixels]
    second = [camera.normalize(pixel) for pixel in scene.second_pixels]
    relative = relative_pose_from_essential(essential_matrix(first, second), first, second)
    truth = scene.second_pose  # the first camera is the world frame

    triangulated = [
        triangulate_point(Pose.identity(), relative, n1, n2)
        for n1, n2 in zip(first, second)
    ]
    # Two views fix the scene only up to scale; align it using the true baseline.
    scale = norm(truth.translation)
    errors = [
        norm([scale * a - b for a, b in zip(estimate, point)])
        for estimate, point in zip(triangulated, scene.points)
    ]
    return {
        "rotation_error_deg": rotation_angle_degrees(relative.rotation, truth.rotation),
        "translation_direction_error_deg": direction_error_degrees(
            relative.translation,
            truth.translation,
        ),
        "median_point_error": sorted(errors)[len(errors) // 2],
        "mean_scene_depth": sum(point[2] for point in scene.points) / len(scene.points),
    }


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
    scene = synthetic_scene(points=args.points, noise=args.noise, seed=args.seed)
    absolute = absolute_pose_report(scene, iterations=args.iterations)
    relative = relative_pose_report(scene)

    print("NanoGrad camera pose demo")
    print(f"points: {args.points}, pixel noise: {args.noise} px, seed: {args.seed}")
    print()
    print("second camera view:")
    print(ascii_view(scene.camera, scene.second_pixels))
    print()
    print("1) absolute pose (PnP) of the second camera")
    print(f"   linear DLT:   rotation err {absolute['linear_rotation_error_deg']:.3f} deg, "
          f"translation err {absolute['linear_translation_error']:.4f}, "
          f"reprojection {absolute['linear_rmse_px']:.3f} px")
    for step in absolute["history"]:
        print(f"   LM iter {step.iteration:>2}:  reprojection {step.rmse:.4f} px  (lambda {step.damping:.0e})")
    print(f"   refined:      rotation err {absolute['refined_rotation_error_deg']:.3f} deg, "
          f"translation err {absolute['refined_translation_error']:.4f}")
    print(f"   ground truth reprojection with this noise: {absolute['truth_rmse_px']:.4f} px")
    print()
    print("2) relative pose from matches only (essential matrix)")
    print(f"   rotation err {relative['rotation_error_deg']:.3f} deg, "
          f"translation direction err {relative['translation_direction_error_deg']:.3f} deg")
    print(f"   triangulated points: median error {relative['median_point_error']:.4f} "
          f"(scene depth ~{relative['mean_scene_depth']:.1f})")
    return {"absolute": absolute, "relative": relative}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--points", type=int, default=40)
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args(argv)
    if args.points < 8:
        parser.error("--points must be at least 8")
    if args.noise < 0.0:
        parser.error("--noise must be non-negative")
    return args


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
