import math
import random
import unittest

from geometry import essential_matrix, epipolar_matrix, rotation_angle_degrees, skew
from linalg import mat_mul
from pose_demo import synthetic_scene
from ransac import (
    ransac,
    ransac_pose_dlt,
    ransac_relative_pose,
    required_iterations,
    sampson_distance_sq,
)


class RansacTests(unittest.TestCase):
    def test_required_iterations(self) -> None:
        self.assertEqual(required_iterations(1.0, 8, 0.99), 1)
        self.assertEqual(required_iterations(0.5, 1, 0.75), 2)
        self.assertGreater(required_iterations(0.5, 8, 0.999), 1000)

    def test_generic_ransac_fits_a_line_through_outliers(self) -> None:
        rng = random.Random(0)
        points = [(x, 2.0 * x + 1.0 + rng.gauss(0.0, 0.01)) for x in range(30)]
        points += [(rng.uniform(0, 30), rng.uniform(-50, 50)) for _ in range(15)]

        def fit(indices):
            (x1, y1), (x2, y2) = points[indices[0]], points[indices[-1]]
            if x1 == x2:
                raise ValueError("degenerate sample")
            slope = (y2 - y1) / (x2 - x1)
            return slope, y1 - slope * x1

        def errors(model):
            slope, offset = model
            return [abs(y - (slope * x + offset)) for x, y in points]

        result = ransac(len(points), 2, fit, errors, 0.1, rng=random.Random(1))

        self.assertAlmostEqual(result.model[0], 2.0, places=2)
        self.assertEqual(sorted(result.inliers)[:30], list(range(30)))

    def test_ransac_needs_enough_data(self) -> None:
        with self.assertRaises(ValueError):
            ransac(3, 4, lambda indices: None, lambda model: [], 1.0)

    def test_sampson_distance_is_zero_for_exact_matches(self) -> None:
        scene = synthetic_scene(points=10, noise=0.0, seed=2)
        camera = scene.camera
        truth = mat_mul(skew(scene.second_pose.translation), scene.second_pose.rotation)

        for first, second in zip(scene.first_pixels, scene.second_pixels):
            distance = sampson_distance_sq(truth, camera.normalize(first), camera.normalize(second))
            self.assertAlmostEqual(distance, 0.0)

    def test_rank_two_model_fits_noisy_matches_better_than_essential(self) -> None:
        scene = synthetic_scene(points=40, noise=0.5, seed=3)
        first = [scene.camera.normalize(pixel) for pixel in scene.first_pixels]
        second = [scene.camera.normalize(pixel) for pixel in scene.second_pixels]

        def mean_distance(matrix):
            return sum(
                math.sqrt(sampson_distance_sq(matrix, a, b)) for a, b in zip(first, second)
            ) / len(first)

        self.assertLess(mean_distance(epipolar_matrix(first, second)), mean_distance(essential_matrix(first, second)))

    def test_robust_estimators_reject_planted_wrong_matches(self) -> None:
        scene = synthetic_scene(points=40, noise=0.5, outliers=0.3, seed=4)
        camera = scene.camera
        first = [camera.normalize(pixel) for pixel in scene.first_pixels]
        second = [camera.normalize(pixel) for pixel in scene.second_pixels]
        threshold = 2.0 / camera.fx

        pnp = ransac_pose_dlt(scene.points, second, threshold=threshold)
        pose, two_view = ransac_relative_pose(first, second, threshold=threshold)

        planted = set(scene.outliers)
        self.assertEqual(set(pnp.inliers) & planted, set())
        self.assertEqual(len(pnp.inliers), 28)
        self.assertLess(rotation_angle_degrees(pnp.model.rotation, scene.second_pose.rotation), 0.5)
        # An epipolar test cannot reject a wrong match that happens to lie on
        # its epipolar line, so allow one through.
        self.assertLessEqual(len(set(two_view.inliers) & planted), 1)
        self.assertGreaterEqual(len(set(two_view.inliers) - planted), 26)
        self.assertLess(rotation_angle_degrees(pose.rotation, scene.second_pose.rotation), 3.0)


if __name__ == "__main__":
    unittest.main()
