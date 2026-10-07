"""From two images to camera motion: corners, matches, RANSAC, bundle adjustment.

The images are rendered from a small textured 3D scene (a back wall, a
floor, and a side wall made of random gray tiles), so the true camera motion
and the true 3D position of every pixel are known. Everything after
rendering only sees pixels:

1. Harris corners in both images (NanoGrad convolutions)
2. NCC patch matching, mutual best + ratio test
3. RANSAC two-view geometry -> relative pose
4. triangulation + two-view bundle adjustment

Ground truth is then used to grade each stage, e.g. which matches were
actually correct.

    python3 image_pair_demo.py
    python3 image_pair_demo.py --output-dir /tmp/pair   # writes PGM images

Rendering is plain ray casting in Python and takes a second or two.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

from bundle_adjustment import two_view_bundle_adjustment
from features import Corner, harris_corners, match_descriptors, patch_descriptor, save_pgm
from geometry import PinholeCamera, Pose, rotation_angle_degrees, triangulate_matches
from linalg import Vector, dot, norm, normalize, transpose
from ransac import ransac_relative_pose
from tensor import Tensor


CAMERA = PinholeCamera(fx=220.0, fy=220.0, cx=160.0, cy=120.0, width=320, height=240)
SECOND_POSE = Pose.from_axis_angle([0.02, -0.12, 0.01], [-0.6, 0.05, 0.05])


# ---------------------------------------------------------------------------
# A tiny ray-cast renderer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TiledPlane:
    """A plane n . X = offset, painted with random gray tiles.

    ``u_axis`` and ``v_axis`` span the plane; a point's texture coordinates
    are its dot products with them, measured in tiles of ``tile`` size. The
    tile grid is turned by ``angle`` radians and every row of tiles is
    shifted by a random amount (like bricks laid carelessly).

    Why bother: on a straight, aligned grid the tile corners sit on long
    straight lines. When those lines run along the epipolar lines, a corner
    matched to its neighbour in the same row still passes the epipolar test,
    and two views cannot reject it. Repetitive structure like this is a
    classic failure mode of feature matching.
    """

    normal: Vector
    offset: float
    u_axis: Vector
    v_axis: Vector
    tile: float
    seed: int
    angle: float = 0.5

    def shade(self, point: Vector) -> float:
        a = dot(point, self.u_axis) / self.tile
        b = dot(point, self.v_axis) / self.tile
        cos_angle, sin_angle = math.cos(self.angle), math.sin(self.angle)
        u = cos_angle * a - sin_angle * b
        v = sin_angle * a + cos_angle * b
        row = math.floor(v)
        column = math.floor(u + _tile_noise(self.seed + 1, row, 0))
        return 0.15 + 0.7 * _tile_noise(self.seed, column, row)


def _tile_noise(seed: int, u: int, v: int) -> float:
    """A deterministic pseudo-random number in [0, 1) per tile.

    Integer hashing (multiply by large odd constants, xor-shift to mix high
    bits into low ones) is much cheaper than seeding a random generator for
    every ray.
    """

    h = (u * 73856093) ^ (v * 19349663) ^ (seed * 83492791)
    h = (h * 2654435761) & 0xFFFFFFFF
    h ^= h >> 15
    h = (h * 2246822519) & 0xFFFFFFFF
    h ^= h >> 13
    return h / 2**32


def default_scene() -> list[TiledPlane]:
    return [
        TiledPlane([0.0, 0.0, 1.0], 7.0, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.45, 1),   # back wall
        TiledPlane([0.0, 1.0, 0.0], 1.4, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.5, 2),    # floor
        TiledPlane([1.0, 0.0, 0.0], -2.6, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 0.4, 3),   # left wall
    ]


def cast_ray(planes: list[TiledPlane], origin: Vector, direction: Vector) -> tuple[float, TiledPlane] | None:
    """Nearest plane hit along origin + s * direction (s > 0)."""

    best: tuple[float, TiledPlane] | None = None
    for plane in planes:
        facing = dot(plane.normal, direction)
        if abs(facing) < 1e-9:
            continue
        distance = (plane.offset - dot(plane.normal, origin)) / facing
        if distance > 1e-6 and (best is None or distance < best[0]):
            best = (distance, plane)
    return best


def render(
    camera: PinholeCamera,
    pose: Pose,
    planes: list[TiledPlane],
    *,
    samples: int = 2,
    noise: float = 0.01,
    seed: int = 0,
) -> Tensor:
    """Ray-cast an image: ``samples`` x ``samples`` rays per pixel, averaged.

    The ray through normalized image point (x, y) has world direction
    R^T [x, y, 1] = x * a + y * b + c, where a, b, c are the columns of R^T,
    so each ray costs a few multiply-adds instead of a matrix product.
    """

    rng = random.Random(seed)
    a, b, c = (list(column) for column in zip(*transpose(pose.rotation)))
    origin = pose.center
    data = []
    for row in range(camera.height):
        for col in range(camera.width):
            total = 0.0
            for sub_row in range(samples):
                y = (row + (sub_row + 0.5) / samples - camera.cy) / camera.fy
                for sub_col in range(samples):
                    x = (col + (sub_col + 0.5) / samples - camera.cx) / camera.fx
                    direction = [x * a[i] + y * b[i] + c[i] for i in range(3)]
                    hit = cast_ray(planes, origin, direction)
                    if hit is not None:
                        distance, plane = hit
                        point = [o + distance * d for o, d in zip(origin, direction)]
                        total += plane.shade(point)
            value = total / (samples * samples) + rng.gauss(0.0, noise)
            data.append(min(1.0, max(0.0, value)))
    return Tensor(data, (camera.height, camera.width))


def pixel_center(corner: Corner) -> Vector:
    """Image position of a corner's pixel center.

    Pixel (col, row) covers [col, col + 1) x [row, row + 1) in the renderer's
    image coordinates, so its center is at (col + 0.5, row + 0.5).
    """

    return [corner.x + 0.5, corner.y + 0.5]


def true_point(planes: list[TiledPlane], camera: PinholeCamera, pixel: Vector) -> Vector:
    """3D point seen at ``pixel`` by the first camera (at the world origin)."""

    direction = [(pixel[0] - camera.cx) / camera.fx, (pixel[1] - camera.cy) / camera.fy, 1.0]
    hit = cast_ray(planes, [0.0, 0.0, 0.0], direction)
    assert hit is not None
    return [hit[0] * value for value in direction]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    *,
    camera: PinholeCamera = CAMERA,
    max_corners: int = 250,
    patch_radius: int = 7,
    threshold_px: float = 1.5,
    huber_px: float = 0.5,
    seed: int = 0,
    output_dir: Path | None = None,
) -> dict:
    planes = default_scene()
    first_image = render(camera, Pose.identity(), planes, seed=seed)
    second_image = render(camera, SECOND_POSE, planes, seed=seed + 1)

    first_corners = harris_corners(first_image, max_corners=max_corners)
    second_corners = harris_corners(second_image, max_corners=max_corners)
    matches = match_descriptors(
        [patch_descriptor(first_image, corner, radius=patch_radius) for corner in first_corners],
        [patch_descriptor(second_image, corner, radius=patch_radius) for corner in second_corners],
    )
    first_pixels = [pixel_center(first_corners[m.first]) for m in matches]
    second_pixels = [pixel_center(second_corners[m.second]) for m in matches]

    # Grade matches with ground truth: project the true 3D point seen at the
    # first corner into the second image and measure the distance.
    match_errors = []
    for first, second in zip(first_pixels, second_pixels):
        projected = camera.project(SECOND_POSE.apply(true_point(planes, camera, first)))
        match_errors.append(math.dist(projected, second) if projected else math.inf)
    correct = [error < 2.0 for error in match_errors]

    first_normalized = [camera.normalize(pixel) for pixel in first_pixels]
    second_normalized = [camera.normalize(pixel) for pixel in second_pixels]
    relative, robust = ransac_relative_pose(
        first_normalized,
        second_normalized,
        threshold=threshold_px / camera.fx,
        rng=random.Random(seed),
    )
    triangulated, kept = triangulate_matches(
        relative,
        [first_normalized[i] for i in robust.inliers],
        [second_normalized[i] for i in robust.inliers],
    )
    inliers = [robust.inliers[k] for k in kept]   # RANSAC inliers in front of both cameras
    adjusted = two_view_bundle_adjustment(
        camera,
        relative,
        triangulated,
        [first_pixels[i] for i in inliers],
        [second_pixels[i] for i in inliers],
        huber_px=huber_px,
        reject_px=threshold_px,
    )
    inliers = [inliers[k] for k in adjusted.kept]

    # Depth check: scale the reconstruction by the true baseline length.
    scale = norm(SECOND_POSE.translation)
    depth_errors = sorted(
        abs(scale * point[2] - true_point(planes, camera, first_pixels[i])[2])
        / true_point(planes, camera, first_pixels[i])[2]
        for point, i in zip(adjusted.points, inliers)
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_pgm(output_dir / "first.pgm", first_image)
        save_pgm(output_dir / "second.pgm", second_image)
        save_pgm(output_dir / "first_inliers.pgm", _mark(first_image, [first_pixels[i] for i in inliers]))
        save_pgm(output_dir / "second_inliers.pgm", _mark(second_image, [second_pixels[i] for i in inliers]))

    def direction_error(estimate: Pose) -> float:
        cosine = dot(normalize(estimate.translation), normalize(SECOND_POSE.translation))
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    return {
        "corners": (len(first_corners), len(second_corners)),
        "matches": len(matches),
        "correct_matches": sum(correct),
        "inliers": len(robust.inliers),
        "correct_inliers": sum(correct[i] for i in robust.inliers),
        "triangulated": len(triangulated),
        "kept_after_ba": len(inliers),
        "correct_after_ba": sum(correct[i] for i in inliers),
        "ransac_iterations": robust.iterations,
        "ransac_rotation_error_deg": rotation_angle_degrees(relative.rotation, SECOND_POSE.rotation),
        "ransac_direction_error_deg": direction_error(relative),
        "ba_rotation_error_deg": rotation_angle_degrees(adjusted.pose.rotation, SECOND_POSE.rotation),
        "ba_direction_error_deg": direction_error(adjusted.pose),
        "ba_rmse_px": adjusted.history[-1].rmse,
        "median_depth_error": depth_errors[len(depth_errors) // 2],
    }


def _mark(image: Tensor, pixels: list[Vector]) -> Tensor:
    """Copy of the image with a small white cross on each pixel."""

    height, width = image.shape
    data = image.data[:]
    for x, y in pixels:
        col, row = int(x), int(y)
        for d in range(-3, 4):
            for r, c in ((row + d, col), (row, col + d)):
                if 0 <= r < height and 0 <= c < width:
                    data[r * width + c] = 1.0
    return Tensor(data, (height, width))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--max-corners", type=int, default=250)
    parser.add_argument("--threshold", type=float, default=1.5, help="RANSAC threshold in px")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    report = run_pipeline(
        max_corners=args.max_corners,
        threshold_px=args.threshold,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print("NanoGrad image-pair demo (rendered 320x240 scene, ground truth known)")
    print(f"Harris corners:      {report['corners'][0]} / {report['corners'][1]}")
    print(f"NCC matches:         {report['matches']} ({report['correct_matches']} correct within 2 px)")
    print(f"RANSAC inliers:      {report['inliers']} ({report['correct_inliers']} correct, {report['triangulated']} in front of both cameras), "
          f"{report['ransac_iterations']} iterations")
    print(f"RANSAC pose:         rotation err {report['ransac_rotation_error_deg']:.3f} deg, "
          f"translation direction err {report['ransac_direction_error_deg']:.2f} deg")
    print(f"BA outlier rejection: kept {report['kept_after_ba']} "
          f"({report['correct_after_ba']} correct)")
    print(f"bundle adjusted:     rotation err {report['ba_rotation_error_deg']:.3f} deg, "
          f"translation direction err {report['ba_direction_error_deg']:.2f} deg, "
          f"reprojection {report['ba_rmse_px']:.3f} px")
    print(f"reconstructed depth: median error {report['median_depth_error']:.1%} (true scale)")
    if args.output_dir is not None:
        print(f"images written to {args.output_dir}")


if __name__ == "__main__":
    main()
