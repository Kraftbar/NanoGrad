import math
import unittest

from geometry import PinholeCamera, Pose
from image_pair_demo import CAMERA, SECOND_POSE, default_scene, link_is_correct, true_point, true_point_seen
from three_view_demo import THIRD_POSE, run_three_views


class ThreeViewTests(unittest.TestCase):
    def test_true_point_seen_matches_first_camera_helper(self) -> None:
        planes = default_scene()
        pixel = [100.5, 80.5]

        from_origin = true_point_seen(planes, CAMERA, Pose.identity(), pixel)

        for a, b in zip(from_origin, true_point(planes, CAMERA, pixel)):
            self.assertAlmostEqual(a, b)

    def test_true_point_seen_reprojects_into_its_own_camera(self) -> None:
        planes = default_scene()
        pixel = [200.25, 60.75]

        point = true_point_seen(planes, CAMERA, THIRD_POSE, pixel)
        projected = CAMERA.project(THIRD_POSE.apply(point))

        self.assertLess(math.dist(projected, pixel), 1e-9)

    def test_link_is_correct(self) -> None:
        planes = default_scene()
        pixel = [150.0, 100.0]
        point = true_point_seen(planes, CAMERA, SECOND_POSE, pixel)
        in_third = CAMERA.project(THIRD_POSE.apply(point))

        self.assertTrue(link_is_correct(planes, CAMERA, SECOND_POSE, THIRD_POSE, pixel, in_third))
        moved = [in_third[0] + 5.0, in_third[1]]
        self.assertFalse(link_is_correct(planes, CAMERA, SECOND_POSE, THIRD_POSE, pixel, moved))

    def test_small_three_view_reconstruction(self) -> None:
        camera = PinholeCamera(fx=110.0, fy=110.0, cx=80.0, cy=60.0, width=160, height=120)

        report = run_three_views(camera=camera, max_corners=150, seed=0)

        self.assertGreater(report["tracks"], 20)
        self.assertGreater(report["pnp_inliers"], 0.8 * report["tracks"])
        self.assertEqual(report["final_wrong"], 0)
        self.assertLess(report["second_rotation_error_deg"], 1.0)
        self.assertLess(report["third_rotation_error_deg"], 1.0)
        # Camera 3 shares camera 2's scale, so its full translation is checked.
        self.assertLess(report["third_translation_error"], 0.08)
        self.assertLess(report["median_depth_error"], 0.05)


if __name__ == "__main__":
    unittest.main()
