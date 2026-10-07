import random
import unittest

from bundle_adjustment import _schur_step, _linearize_point, two_view_bundle_adjustment, two_view_rmse
from geometry import Pose, apply_pose_update, rotation_angle_degrees
from linalg import mat_mul, mat_vec, norm, solve, transpose
from pose_demo import synthetic_scene


class BundleAdjustmentTests(unittest.TestCase):
    def test_noise_free_problem_converges_to_truth(self) -> None:
        scene = synthetic_scene(points=20, noise=0.0, seed=1)
        rng = random.Random(0)
        start_pose = apply_pose_update(scene.second_pose, [0.02, -0.01, 0.015, 0.05, -0.03, 0.02])
        start_points = [[value + rng.gauss(0.0, 0.1) for value in point] for point in scene.points]

        result = two_view_bundle_adjustment(
            scene.camera,
            start_pose,
            start_points,
            scene.first_pixels,
            scene.second_pixels,
            iterations=30,
        )

        self.assertGreater(result.history[0].rmse, 5.0)
        self.assertLess(result.history[-1].rmse, 1e-8)
        self.assertLess(rotation_angle_degrees(result.pose.rotation, scene.second_pose.rotation), 1e-6)

    def test_scale_is_held_fixed(self) -> None:
        scene = synthetic_scene(points=15, noise=0.5, seed=2)
        start_pose = apply_pose_update(scene.second_pose, [0.01, 0.0, 0.0, 0.1, 0.0, 0.0])

        result = two_view_bundle_adjustment(
            scene.camera,
            start_pose,
            scene.points,
            scene.first_pixels,
            scene.second_pixels,
        )

        self.assertAlmostEqual(norm(result.pose.translation), norm(start_pose.translation))
        for earlier, later in zip(result.history, result.history[1:]):
            self.assertLessEqual(later.rmse, earlier.rmse)

    def test_schur_step_matches_full_normal_equations(self) -> None:
        scene = synthetic_scene(points=4, noise=0.5, seed=3)
        pose = apply_pose_update(scene.second_pose, [0.01, 0.02, 0.0, 0.0, 0.05, 0.0])
        blocks = [
            _linearize_point(scene.camera, pose, point, first, second)
            for point, first, second in zip(scene.points, scene.first_pixels, scene.second_pixels)
        ]
        damping = 0.1

        pose_step, point_steps = _schur_step(blocks, damping)

        # Dense (6 + 3N) system with the same Marquardt damping.
        size = 6 + 3 * len(blocks)
        rows = []
        residuals = []
        for index, block in enumerate(blocks):
            for row, (pose_row, point_row) in enumerate(zip(block.pose_jacobian, block.point_jacobian)):
                full = [0.0] * size
                full[:6] = pose_row
                full[6 + 3 * index : 9 + 3 * index] = point_row
                rows.append(full)
                residuals.append(block.residuals[row])
        rows_t = transpose(rows)
        hessian = mat_mul(rows_t, rows)
        for i in range(size):
            hessian[i][i] *= 1.0 + damping
        dense = solve(hessian, [-value for value in mat_vec(rows_t, residuals)])

        expected = [*pose_step, *[value for step in point_steps for value in step]]
        for a, b in zip(dense, expected):
            self.assertAlmostEqual(a, b, places=8)

    def test_rejection_refits_after_the_last_prune(self) -> None:
        scene = synthetic_scene(points=30, noise=0.0, seed=5)
        second = [pixel[:] for pixel in scene.second_pixels]
        for index in (0, 1, 2):
            second[index] = [second[index][0] + 25.0, second[index][1] - 15.0]

        result = two_view_bundle_adjustment(
            scene.camera,
            scene.second_pose,
            scene.points,
            scene.first_pixels,
            second,
            huber_px=0.5,
            reject_px=1.0,
            rounds=2,
        )

        self.assertEqual(result.kept, list(range(3, 30)))
        kept_rmse = two_view_rmse(
            scene.camera,
            result.pose,
            result.points,
            [scene.first_pixels[i] for i in result.kept],
            [second[i] for i in result.kept],
        )
        self.assertAlmostEqual(result.history[-1].rmse, kept_rmse)
        self.assertLess(kept_rmse, 1e-6)
        iterations = [step.iteration for step in result.history]
        self.assertEqual(iterations, sorted(set(iterations)))

    def test_needs_points(self) -> None:
        with self.assertRaises(ValueError):
            camera = synthetic_scene(points=8).camera
            two_view_bundle_adjustment(camera, Pose.identity(), [], [], [])

    def test_rmse_counts_both_images(self) -> None:
        scene = synthetic_scene(points=5, noise=0.0, seed=4)
        shifted = [[u + 3.0, v + 4.0] for u, v in scene.second_pixels]

        rmse = two_view_rmse(scene.camera, scene.second_pose, scene.points, scene.first_pixels, shifted)

        # Half the residuals are 5 px, half are 0.
        self.assertAlmostEqual(rmse, (25.0 / 2) ** 0.5)
        self.assertEqual(Pose.identity().translation, [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
