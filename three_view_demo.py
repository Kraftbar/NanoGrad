"""A third view: tracks, PnP, and three-view bundle adjustment.

Two views have a blind spot: a wrong match that happens to lie on its
epipolar line triangulates to a wrong but self-consistent 3D point, and no
two-view test can catch it. A third camera can, because that wrong 3D point
projects to the wrong place in the new image.

Pipeline (images rendered as in image_pair_demo.py, ground truth known):

1. two-view reconstruction from images 1 and 2 (RANSAC + bundle adjustment)
2. match image 2 to image 3 and chain the matches into tracks 1 -> 2 -> 3
3. RANSAC PnP: where is camera 3, given the 3D points from step 1?
4. tracks whose third observation disagrees with PnP are dropped (either
   link could be the wrong one, and there are plenty of good tracks)
5. bundle adjustment of all three cameras and all points together

Because camera 3 is placed relative to points whose scale was fixed by
cameras 1 and 2, all three cameras share one scale. Two separate pairs
would each have their own.

    python3 three_view_demo.py
"""

from __future__ import annotations

import argparse
import math
import random

from bundle_adjustment import Observation, bundle_adjust
from geometry import PinholeCamera, Pose, refine_pose, rotation_angle_degrees
from image_pair_demo import (
    CAMERA,
    SECOND_POSE,
    TiledPlane,
    cast_ray,
    default_scene,
    detect,
    direction_error_degrees,
    match_features,
    pixel_center,
    reconstruct_two_view,
    render,
)
from linalg import Vector, norm
from ransac import ransac_pose_dlt


THIRD_POSE = Pose.from_axis_angle([0.03, -0.24, 0.02], [-1.2, 0.08, 0.15])


def true_point_seen(
    planes: list[TiledPlane],
    camera: PinholeCamera,
    pose: Pose,
    pixel: Vector,
) -> Vector | None:
    """World point seen at ``pixel`` by a camera at ``pose``."""

    ray_camera = [(pixel[0] - camera.cx) / camera.fx, (pixel[1] - camera.cy) / camera.fy, 1.0]
    rotation_t = list(zip(*pose.rotation))
    direction = [sum(rotation_t[row][k] * ray_camera[k] for k in range(3)) for row in range(3)]
    origin = pose.center
    hit = cast_ray(planes, origin, direction)
    if hit is None:
        return None
    return [o + hit[0] * d for o, d in zip(origin, direction)]


def link_is_correct(
    planes: list[TiledPlane],
    camera: PinholeCamera,
    from_pose: Pose,
    to_pose: Pose,
    from_pixel: Vector,
    to_pixel: Vector,
    *,
    tolerance_px: float = 2.0,
) -> bool:
    point = true_point_seen(planes, camera, from_pose, from_pixel)
    projected = None if point is None else camera.project(to_pose.apply(point))
    return projected is not None and math.dist(projected, to_pixel) < tolerance_px


def run_three_views(
    *,
    camera: PinholeCamera = CAMERA,
    max_corners: int = 250,
    threshold_px: float = 1.5,
    huber_px: float = 0.5,
    seed: int = 0,
) -> dict:
    planes = default_scene()
    true_poses = [Pose.identity(), SECOND_POSE, THIRD_POSE]
    images = [render(camera, pose, planes, seed=seed + k) for k, pose in enumerate(true_poses)]
    features = [detect(image, max_corners=max_corners) for image in images]

    # 1. Two-view reconstruction from images 1 and 2.
    matches_12, first_pixels, second_pixels = match_features(features[0], features[1])
    two = reconstruct_two_view(
        camera,
        first_pixels,
        second_pixels,
        threshold_px=threshold_px,
        huber_px=huber_px,
        seed=seed,
    )
    correct_12 = [
        link_is_correct(planes, camera, true_poses[0], true_poses[1], a, b)
        for a, b in zip(first_pixels, second_pixels)
    ]

    # 2. Tracks: follow each kept 1-2 match through its image-2 corner.
    matches_23, _, _ = match_features(features[1], features[2])
    next_corner = {match.first: match.second for match in matches_23}
    tracks = []   # (index into two.points, pixel in image 3)
    for point_index, match_index in enumerate(two.kept):
        corner_2 = matches_12[match_index].second
        if corner_2 in next_corner:
            tracks.append((point_index, pixel_center(features[2].corners[next_corner[corner_2]])))

    # 3. Camera 3 from 3D points and their image-3 pixels.
    track_points = [two.points[point_index] for point_index, _ in tracks]
    track_pixels = [pixel for _, pixel in tracks]
    pnp = ransac_pose_dlt(
        track_points,
        [camera.normalize(pixel) for pixel in track_pixels],
        threshold=threshold_px / camera.fx,
        rng=random.Random(seed),
    )
    third_pose, _ = refine_pose(
        camera,
        pnp.model,
        [track_points[i] for i in pnp.inliers],
        [track_pixels[i] for i in pnp.inliers],
    )

    # 4. Drop tracks that the third view contradicts.
    agreeing = set(pnp.inliers)
    dropped_points = {tracks[i][0] for i in range(len(tracks)) if i not in agreeing}
    observations = []
    for point_index, match_index in enumerate(two.kept):
        if point_index in dropped_points:
            continue
        observations.append(Observation(point_index, 0, first_pixels[match_index]))
        observations.append(Observation(point_index, 1, second_pixels[match_index]))
    for i in pnp.inliers:
        point_index, pixel = tracks[i]
        observations.append(Observation(point_index, 2, pixel))

    # 5. Three-view bundle adjustment.
    adjusted = bundle_adjust(
        camera,
        [Pose.identity(), two.pose, third_pose],
        two.points,
        observations,
        huber_px=huber_px,
        reject_px=threshold_px,
    )
    final_points = adjusted.active_points()

    # Grading. The reconstruction's scale is arbitrary, so measure it once
    # from camera 2 and use that one factor for camera 3 and the depths.
    scale = norm(SECOND_POSE.translation) / norm(adjusted.poses[1].translation)
    third_translation_error = math.dist(
        [scale * value for value in adjusted.poses[2].translation],
        THIRD_POSE.translation,
    ) / norm(THIRD_POSE.translation)
    depth_errors = []
    for point_index in final_points:
        truth = true_point_seen(planes, camera, Pose.identity(), first_pixels[two.kept[point_index]])
        depth_errors.append(abs(scale * adjusted.points[point_index][2] - truth[2]) / truth[2])
    depth_errors.sort()

    wrong_two_view = [k for k, m in enumerate(two.kept) if not correct_12[m]]
    return {
        "matches_12": len(matches_12),
        "two_view_points": len(two.kept),
        "two_view_wrong": len(wrong_two_view),
        "two_view_rotation_error_deg": rotation_angle_degrees(two.pose.rotation, SECOND_POSE.rotation),
        "two_view_direction_error_deg": direction_error_degrees(two.pose.translation, SECOND_POSE.translation),
        "matches_23": len(matches_23),
        "tracks": len(tracks),
        "pnp_inliers": len(pnp.inliers),
        "dropped_by_third_view": len(dropped_points),
        "wrong_dropped": sum(k in dropped_points for k in wrong_two_view),
        "final_points": len(final_points),
        "final_wrong": sum(k in final_points for k in wrong_two_view),
        "second_rotation_error_deg": rotation_angle_degrees(adjusted.poses[1].rotation, SECOND_POSE.rotation),
        "second_direction_error_deg": direction_error_degrees(adjusted.poses[1].translation, SECOND_POSE.translation),
        "third_rotation_error_deg": rotation_angle_degrees(adjusted.poses[2].rotation, THIRD_POSE.rotation),
        "third_translation_error": third_translation_error,
        "rmse_px": adjusted.history[-1].rmse,
        "median_depth_error": depth_errors[len(depth_errors) // 2],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--max-corners", type=int, default=250)
    parser.add_argument("--threshold", type=float, default=1.5, help="RANSAC threshold in px")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    r = run_three_views(max_corners=args.max_corners, threshold_px=args.threshold, seed=args.seed)
    print("NanoGrad three-view demo (rendered 320x240 scene, ground truth known)")
    print(f"images 1-2: {r['matches_12']} matches -> {r['two_view_points']} reconstructed points "
          f"({r['two_view_wrong']} from wrong matches)")
    print(f"  two-view pose 2:  rotation err {r['two_view_rotation_error_deg']:.3f} deg, "
          f"translation direction err {r['two_view_direction_error_deg']:.2f} deg")
    print(f"images 2-3: {r['matches_23']} matches -> {r['tracks']} tracks through all three images")
    print(f"PnP for camera 3: {r['pnp_inliers']} of {r['tracks']} tracks agree; "
          f"{r['dropped_by_third_view']} dropped ({r['wrong_dropped']} were wrong 1-2 matches)")
    print(f"three-view BA: {r['final_points']} points ({r['final_wrong']} from wrong matches), "
          f"reprojection {r['rmse_px']:.3f} px")
    print(f"  pose 2:  rotation err {r['second_rotation_error_deg']:.3f} deg, "
          f"translation direction err {r['second_direction_error_deg']:.2f} deg")
    print(f"  pose 3:  rotation err {r['third_rotation_error_deg']:.3f} deg, "
          f"translation err {r['third_translation_error']:.1%} (same scale as pose 2)")
    print(f"  depth:   median error {r['median_depth_error']:.1%}")


if __name__ == "__main__":
    main()
