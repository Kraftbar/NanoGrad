import contextlib
import importlib.util
import io
import unittest

from sine import main, sine_waves


class SineTests(unittest.TestCase):
    def test_waves_are_shifted_sines(self) -> None:
        waves = sine_waves(4, 50)
        self.assertEqual(len(waves), 4)
        self.assertTrue(all(-1.0 <= value <= 1.0 for wave in waves for value in wave))
        self.assertNotEqual(waves[0][:5], waves[1][:5])

    def test_lbfgs_fits_and_the_model_runs_free(self) -> None:
        frameworks = ["nanograd", "torch"] if importlib.util.find_spec("torch") else ["nanograd"]
        with contextlib.redirect_stdout(io.StringIO()):
            results = main(["--steps", "1", "--waves", "6", "--length", "60", "--hidden", "6", "--future", "10",
                            "--frameworks", *frameworks])
        first = results["nanograd"]["history"][0]
        self.assertLess(first[1], 0.5)
        self.assertIn("future_mse", results)
        if "torch" in results:
            self.assertAlmostEqual(first[1], results["torch"]["history"][0][1], places=10)


if __name__ == "__main__":
    unittest.main()
