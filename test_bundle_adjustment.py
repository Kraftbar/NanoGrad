import random
import unittest

from bundle_adjustment import (
    Observation,
    _linearize,
    _schur_step,
    bundle_adjust,
    two_view_bundle_adjustment,
    two_view_rmse,
)
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

    def assertSchurMatchesDense(self, poses, points, observations, damping=0.1) -> None:
        camera = synthetic_scene(points=8).camera
        rows = [_linearize(camera, poses, points, observation) for observation in observations]
        active = sorted({observation.point for observation in observations})

        pose_steps, point_steps = _schur_step(rows, active, len(poses), damping)

        # Dense (6 (V - 1) + 3 N) system with the same Marquardt damping.
        cameras = 6 * (len(poses) - 1)
        size = cameras + 3 * len(active)
        column_of = {index: cameras + 3 * k for k, index in enumerate(active)}
        dense_rows = []
        residuals = []
        for row in rows:
            for r in range(2):
                full = [0.0] * size
                if row.pose_jacobian is not None:
                    start = 6 * (row.view - 1)
                    full[start : start + 6] = row.pose_jacobian[r]
                full[column_of[row.point] : column_of[row.point] + 3] = row.point_jacobian[r]
                dense_rows.append(full)
                residuals.append(row.residuals[r])
        rows_t = transpose(dense_rows)
        hessian = mat_mul(rows_t, dense_rows)
        for i in range(size):
            hessian[i][i] *= 1.0 + damping
        dense = solve(hessian, [-value for value in mat_vec(rows_t, residuals)])

        expected = [
            *[value for step in pose_steps for value in step],
            *[value for index in active for value in point_steps[index]],
        ]
        for a, b in zip(dense, expected):
            self.assertAlmostEqual(a, b, places=8)

    def test_schur_step_matches_full_normal_equations(self) -> None:
        scene = synthetic_scene(points=4, noise=0.5, seed=3)
        pose = apply_pose_update(scene.second_pose, [0.01, 0.02, 0.0, 0.0, 0.05, 0.0])
        observations = [
            Observation(index, view, pixel)
            for index, (first, second) in enumerate(zip(scene.first_pixels, scene.second_pixels))
            for view, pixel in ((0, first), (1, second))
        ]

        self.assertSchurMatchesDense([Pose.identity(), pose], scene.points, observations)

    def test_schur_step_couples_cameras_that_share_points(self) -> None:
        scene = synthetic_scene(points=5, noise=0.5, seed=6)
        third = Pose.from_axis_angle([0.0, 0.1, 0.0], [-1.2, 0.0, 0.1])
        poses = [Pose.identity(), scene.second_pose, third]
        observations = []
        for index, point in enumerate(scene.points):
            for view, pose in enumerate(poses):
                if index == 0 and view == 2:
                    continue  # one point is only seen by two cameras
                observations.append(Observation(index, view, scene.camera.project(pose.apply(point))))
        nudged = [apply_pose_update(pose, [0.01, -0.01, 0.0, 0.02, 0.0, 0.0]) for pose in poses[1:]]

        self.assertSchurMatchesDense([poses[0], *nudged], scene.points, observations)

    def test_three_views_converge_to_truth(self) -> None:
        scene = synthetic_scene(points=15, noise=0.0, seed=7)
        third = Pose.from_axis_angle([0.02, 0.1, 0.0], [-1.2, 0.0, 0.1])
        poses = [Pose.identity(), scene.second_pose, third]
        observations = [
            Observation(index, view, scene.camera.project(pose.apply(point)))
            for index, point in enumerate(scene.points)
            for view, pose in enumerate(poses)
        ]
        rng = random.Random(1)
        start = [
            poses[0],
            apply_pose_update(poses[1], [0.01, 0.0, -0.01, 0.03, 0.0, 0.0]),
            apply_pose_update(poses[2], [0.0, 0.02, 0.01, 0.0, 0.05, -0.02]),
        ]
        noisy = [[value + rng.gauss(0.0, 0.05) for value in point] for point in scene.points]

        result = bundle_adjust(scene.camera, start, noisy, observations)

        self.assertLess(result.history[-1].rmse, 1e-8)
        for estimate, truth in zip(result.poses[1:], poses[1:]):
            self.assertLess(rotation_angle_degrees(estimate.rotation, truth.rotation), 1e-6)
        self.assertAlmostEqual(norm(result.poses[1].translation), norm(start[1].translation))

    def test_observation_level_rejection_keeps_the_rest_of_a_track(self) -> None:
        scene = synthetic_scene(points=20, noise=0.0, seed=8)
        third = Pose.from_axis_angle([0.0, 0.1, 0.0], [-1.2, 0.0, 0.1])
        poses = [Pose.identity(), scene.second_pose, third]
        observations = []
        for index, point in enumerate(scene.points):
            for view, pose in enumerate(poses):
                pixel = scene.camera.project(pose.apply(point))
                if index < 2 and view == 2:
                    pixel = [pixel[0] + 30.0, pixel[1] - 20.0]   # a bad third-view link
                observations.append(Observation(index, view, pixel))

        result = bundle_adjust(
            scene.camera, poses, scene.points, observations, huber_px=0.5, reject_px=1.0,
        )

        dropped = set((o.point, o.view) for o in observations) - set((o.point, o.view) for o in result.observations)
        self.assertEqual(dropped, {(0, 2), (1, 2)})
        self.assertIn(0, result.active_points())
        self.assertLess(result.history[-1].rmse, 1e-6)

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

    def test_rejects_undetermined_cameras_and_same_view_duplicates(self) -> None:
        scene = synthetic_scene(points=10, noise=0.0, seed=9)
        third = Pose.from_axis_angle([0.0, 0.1, 0.0], [-1.2, 0.0, 0.1])
        two_views = [
            Observation(index, view, scene.camera.project(pose.apply(point)))
            for index, point in enumerate(scene.points)
            for view, pose in enumerate([Pose.identity(), scene.second_pose])
        ]

        with self.assertRaises(ValueError):
            bundle_adjust(scene.camera, [Pose.identity(), scene.second_pose, third], scene.points, two_views)

        # Two observations of point 0 in the same view do not make it active.
        duplicate = [Observation(0, 0, [100.0, 100.0]), Observation(0, 0, [101.0, 100.0])]
        result = bundle_adjust(
            scene.camera, [Pose.identity(), scene.second_pose], scene.points, two_views[2:] + duplicate,
        )
        self.assertNotIn(0, result.active_points())

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
