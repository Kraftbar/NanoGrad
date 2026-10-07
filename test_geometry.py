import math
import unittest
from contextlib import redirect_stdout
from io import StringIO

from engine import Value, gradients
from geometry import (
    PinholeCamera,
    Pose,
    axis_angle_from_rotation,
    essential_matrix,
    estimate_pose_dlt,
    refine_pose,
    relative_pose_from_essential,
    reprojection_rmse,
    rotation_angle_degrees,
    rotation_from_axis_angle,
    skew,
    triangulate_point,
)
from linalg import determinant3, identity, mat_mul, mat_vec, transpose
from pose_demo import (
    absolute_pose_report,
    main,
    relative_pose_report,
    synthetic_scene,
)


CAMERA = PinholeCamera(fx=400.0, fy=420.0, cx=160.0, cy=120.0, width=320, height=240)


class RotationTests(unittest.TestCase):
    def assertMatrixAlmostEqual(self, left, right, places=9) -> None:
        for left_row, right_row in zip(left, right):
            for a, b in zip(left_row, right_row):
                self.assertAlmostEqual(a, b, places=places)

    def test_rodrigues_matches_axis_rotation(self) -> None:
        angle = 0.7
        rotation = rotation_from_axis_angle([0.0, 0.0, angle])

        self.assertMatrixAlmostEqual(
            rotation,
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
        )

    def test_rotations_are_orthonormal_and_invertible(self) -> None:
        for axis_angle in ([0.1, -0.4, 0.25], [1e-10, 0.0, 0.0], [0.0, 3.0, 0.1]):
            rotation = rotation_from_axis_angle(axis_angle)
            self.assertMatrixAlmostEqual(mat_mul(transpose(rotation), rotation), identity(3))
            self.assertAlmostEqual(determinant3(rotation), 1.0)
            recovered = axis_angle_from_rotation(rotation)
            for expected, actual in zip(axis_angle, recovered):
                self.assertAlmostEqual(expected, actual, places=6)

    def test_skew_is_cross_product(self) -> None:
        v = [1.0, 2.0, 3.0]
        w = [-0.5, 0.25, 2.0]

        self.assertEqual(
            mat_vec(skew(v), w),
            [2.0 * 2.0 - 3.0 * 0.25, 3.0 * -0.5 - 1.0 * 2.0, 1.0 * 0.25 - 2.0 * -0.5],
        )

    def test_rodrigues_gradient_is_finite_at_zero(self) -> None:
        delta = [Value(0.0) for _ in range(3)]
        rotation = rotation_from_axis_angle(delta)

        # d R / d w_z at zero is the generator [e_z]x, so R[1][0] has slope 1.
        self.assertEqual(gradients(rotation[1][0], delta), [0.0, 0.0, 1.0])
        self.assertEqual(gradients(rotation[0][1], delta), [0.0, 0.0, -1.0])

    def test_rotation_angle_degrees(self) -> None:
        left = rotation_from_axis_angle([0.0, math.radians(30.0), 0.0])

        self.assertAlmostEqual(rotation_angle_degrees(left, identity(3)), 30.0)


class PoseAndCameraTests(unittest.TestCase):
    def test_pose_inverse_and_compose(self) -> None:
        pose = Pose.from_axis_angle([0.2, -0.1, 0.3], [1.0, 2.0, -0.5])
        point = [0.3, -1.2, 4.0]

        back = pose.inverse().apply(pose.apply(point))
        for expected, actual in zip(point, back):
            self.assertAlmostEqual(expected, actual)

        composed = pose.compose(pose.inverse())
        for row, expected_row in zip(composed.rotation, identity(3)):
            for value, expected in zip(row, expected_row):
                self.assertAlmostEqual(value, expected)
        self.assertEqual(Pose.identity().center, [0.0, 0.0, 0.0])

    def test_project_and_normalize(self) -> None:
        pixel = CAMERA.project([0.5, -0.25, 2.0])

        self.assertEqual(pixel, [400.0 * 0.25 + 160.0, 420.0 * -0.125 + 120.0])
        self.assertEqual(CAMERA.normalize(pixel), [0.25, -0.125])
        self.assertIsNone(CAMERA.project([0.0, 0.0, -1.0]))
        self.assertFalse(CAMERA.contains(None))
        self.assertTrue(CAMERA.contains(pixel))


class EstimatorTests(unittest.TestCase):
    def test_triangulation_recovers_exact_point(self) -> None:
        first = Pose.identity()
        second = Pose.from_axis_angle([0.0, 0.1, 0.0], [-0.5, 0.0, 0.0])
        point = [0.4, -0.3, 5.0]

        n1 = CAMERA.normalize(CAMERA.project(first.apply(point)))
        n2 = CAMERA.normalize(CAMERA.project(second.apply(point)))
        estimate = triangulate_point(first, second, n1, n2)

        for expected, actual in zip(point, estimate):
            self.assertAlmostEqual(expected, actual)

    def test_noise_free_estimators_are_exact(self) -> None:
        scene = synthetic_scene(points=20, noise=0.0, seed=1)
        second = [scene.camera.normalize(pixel) for pixel in scene.second_pixels]
        first = [scene.camera.normalize(pixel) for pixel in scene.first_pixels]

        pose = estimate_pose_dlt(scene.points, second)
        self.assertLess(rotation_angle_degrees(pose.rotation, scene.second_pose.rotation), 1e-6)
        for expected, actual in zip(scene.second_pose.translation, pose.translation):
            self.assertAlmostEqual(expected, actual, places=6)

        essential = essential_matrix(first, second)
        for n1, n2 in zip(first, second):
            epipolar = sum(
                [*n2, 1.0][row] * essential[row][col] * [*n1, 1.0][col]
                for row in range(3)
                for col in range(3)
            )
            self.assertAlmostEqual(epipolar, 0.0)

        relative = relative_pose_from_essential(essential, first, second)
        self.assertLess(
            rotation_angle_degrees(relative.rotation, scene.second_pose.rotation),
            1e-6,
        )
        true_direction = [
            value / math.sqrt(sum(t * t for t in scene.second_pose.translation))
            for value in scene.second_pose.translation
        ]
        for expected, actual in zip(true_direction, relative.translation):
            self.assertAlmostEqual(expected, actual, places=6)

    def test_estimators_need_enough_points(self) -> None:
        with self.assertRaises(ValueError):
            estimate_pose_dlt([[0.0, 0.0, 1.0]] * 5, [[0.0, 0.0]] * 5)
        with self.assertRaises(ValueError):
            essential_matrix([[0.0, 0.0]] * 7, [[0.0, 0.0]] * 7)

    def test_refinement_reaches_noise_floor(self) -> None:
        scene = synthetic_scene(points=30, noise=1.0, seed=2)
        start = Pose.from_axis_angle([0.08, 0.1, 0.0], [-0.6, 0.1, 0.3])

        refined, history = refine_pose(
            scene.camera,
            start,
            scene.points,
            scene.second_pixels,
            iterations=15,
        )

        truth_rmse = reprojection_rmse(
            scene.camera,
            scene.second_pose,
            scene.points,
            scene.second_pixels,
        )
        self.assertGreater(history[0].rmse, 10.0)
        self.assertLessEqual(history[-1].rmse, truth_rmse)
        self.assertLess(rotation_angle_degrees(refined.rotation, scene.second_pose.rotation), 0.5)
        for earlier, later in zip(history, history[1:]):
            self.assertLessEqual(later.rmse, earlier.rmse)


class PoseDemoTests(unittest.TestCase):
    def test_reports_with_noise(self) -> None:
        scene = synthetic_scene(points=40, noise=0.5, seed=0)

        absolute = absolute_pose_report(scene)
        relative = relative_pose_report(scene)

        self.assertLess(absolute["refined_rmse_px"], absolute["linear_rmse_px"] + 1e-12)
        self.assertLess(absolute["refined_rotation_error_deg"], 0.5)
        self.assertLess(relative["rotation_error_deg"], 3.0)
        self.assertLess(relative["translation_direction_error_deg"], 10.0)

    def test_main_prints_report(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            main(["--points", "12", "--noise", "0.2"])

        self.assertIn("absolute pose (PnP)", output.getvalue())
        self.assertIn("essential matrix", output.getvalue())


if __name__ == "__main__":
    unittest.main()
