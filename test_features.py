import math
import tempfile
import unittest
from pathlib import Path

from features import (
    Corner,
    harris_corners,
    load_pgm,
    match_descriptors,
    patch_descriptor,
    save_pgm,
)
from geometry import PinholeCamera
from image_pair_demo import _tile_noise, run_pipeline
from tensor import Tensor


def square_image(size: int = 32, low: int = 10, high: int = 22) -> Tensor:
    """Dark image with one bright axis-aligned square [low, high) x [low, high)."""

    data = [
        0.8 if low <= row < high and low <= col < high else 0.2
        for row in range(size)
        for col in range(size)
    ]
    return Tensor(data, (size, size))


def textured_image(width: int, height: int, *, shift: tuple[int, int] = (0, 0)) -> Tensor:
    """Random blocky texture; ``shift`` moves the content right/down."""

    dx, dy = shift
    data = [
        _tile_noise(7, (col - dx) // 4, (row - dy) // 4)
        for row in range(height)
        for col in range(width)
    ]
    return Tensor(data, (height, width))


class HarrisTests(unittest.TestCase):
    def test_finds_the_four_corners_of_a_square(self) -> None:
        corners = harris_corners(square_image(), max_corners=4)

        self.assertEqual(len(corners), 4)
        # The square's corners are the pixel boundaries at 10 and 22. Harris
        # peaks sit about 1.4 px inside a sharp L corner (the window response
        # is largest there), the same bias in every view of that corner.
        for target in ((10, 10), (10, 22), (22, 10), (22, 22)):
            distance = min(math.dist((c.x + 0.5, c.y + 0.5), target) for c in corners)
            self.assertLess(distance, 2.5)

    def test_flat_image_has_no_corners(self) -> None:
        flat = Tensor([0.5] * 400, (20, 20))
        flat.data[0] = 0.51  # avoid a zero threshold

        self.assertLessEqual(len(harris_corners(flat)), 1)

    def test_min_distance_spreads_corners(self) -> None:
        corners = harris_corners(textured_image(60, 40), max_corners=50, min_distance=8)

        # Spacing is enforced on whole pixels; the sub-pixel refinement that
        # follows moves each corner by at most half a pixel per axis.
        for i, a in enumerate(corners):
            for b in corners[i + 1 :]:
                self.assertGreaterEqual(math.dist((a.x, a.y), (b.x, b.y)), 8.0 - math.sqrt(2.0))


class MatchingTests(unittest.TestCase):
    def test_patch_descriptor_is_normalized_and_skips_borders(self) -> None:
        image = textured_image(30, 30)

        descriptor = patch_descriptor(image, Corner(15, 15, 1.0), radius=3)

        self.assertEqual(len(descriptor), 49)
        self.assertAlmostEqual(sum(descriptor), 0.0)
        self.assertAlmostEqual(sum(value * value for value in descriptor), 1.0)
        self.assertIsNone(patch_descriptor(image, Corner(1, 15, 1.0), radius=3))

    def test_matches_follow_a_pure_image_shift(self) -> None:
        first = textured_image(80, 60)
        second = textured_image(80, 60, shift=(3, 2))
        first_corners = harris_corners(first, max_corners=60)
        second_corners = harris_corners(second, max_corners=60)

        matches = match_descriptors(
            [patch_descriptor(first, corner) for corner in first_corners],
            [patch_descriptor(second, corner) for corner in second_corners],
        )

        self.assertGreater(len(matches), 15)
        for match in matches:
            a = first_corners[match.first]
            b = second_corners[match.second]
            self.assertAlmostEqual(b.x - a.x, 3.0, delta=0.6)
            self.assertAlmostEqual(b.y - a.y, 2.0, delta=0.6)

    def test_pgm_round_trip(self) -> None:
        image = textured_image(9, 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.pgm"
            save_pgm(path, image)
            loaded = load_pgm(path)

        self.assertEqual(loaded.shape, (5, 9))
        for original, restored in zip(image.data, loaded.data):
            self.assertAlmostEqual(original, restored, delta=1 / 255)


class PgmEdgeCaseTests(unittest.TestCase):
    def write(self, directory: str, content: bytes) -> Path:
        path = Path(directory) / "image.pgm"
        path.write_bytes(content)
        return path

    def test_header_comments_and_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = load_pgm(self.write(directory, b"P5 # comment\n2#c\n1 255\n\x00\xff"))
            self.assertEqual(image.data, [0.0, 1.0])
            for bad in (b"P5 2 1 255", b"P5 2 1 0\n\x00\x00", b"P5 2 1 255\n\x00", b"P2 2 1 255\n00"):
                with self.subTest(content=bad), self.assertRaises(ValueError):
                    load_pgm(self.write(directory, bad))

    def test_even_harris_window_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            harris_corners(square_image(), window=4)


class ImagePairPipelineTests(unittest.TestCase):
    def test_small_rendered_pair_recovers_rotation(self) -> None:
        camera = PinholeCamera(fx=110.0, fy=110.0, cx=80.0, cy=60.0, width=160, height=120)

        report = run_pipeline(camera=camera, max_corners=150, seed=0)

        self.assertGreater(report["matches"], 40)
        self.assertGreater(report["correct_matches"] / report["matches"], 0.8)
        self.assertGreaterEqual(report["correct_after_ba"], report["kept_after_ba"] - 1)
        self.assertLess(report["ba_rotation_error_deg"], 2.0)
        self.assertLess(report["ba_direction_error_deg"], 15.0)


if __name__ == "__main__":
    unittest.main()
