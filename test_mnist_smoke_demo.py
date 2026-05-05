import io
import unittest
from contextlib import redirect_stdout

from mnist_smoke_demo import main, smoke_dataset


class MNISTSmokeDemoTests(unittest.TestCase):
    def test_smoke_dataset(self) -> None:
        dataset = smoke_dataset()

        self.assertEqual(len(dataset), 4)
        self.assertEqual(dataset[0], ([1.0, 0.0, 0.0, 0.0], 0.0))
        self.assertEqual(dataset[3], ([0.0, 0.0, 0.1, 0.9], 1.0))

    def test_smoke_demo_runs(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            main()

        text = output.getvalue()
        self.assertIn("MNIST-style smoke demo", text)
        self.assertIn("samples:      4", text)
        self.assertIn("accuracy:     1.000", text)


if __name__ == "__main__":
    unittest.main()
