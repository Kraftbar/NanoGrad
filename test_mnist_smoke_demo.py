import io
import unittest
from contextlib import redirect_stdout

from tensor import Tensor
from mnist_smoke_demo import SmokeCNN, main, smoke_dataset


class MNISTSmokeDemoTests(unittest.TestCase):
    def test_smoke_dataset(self) -> None:
        dataset = smoke_dataset()

        self.assertEqual(len(dataset), 4)
        self.assertEqual(dataset.feature_shape, (1, 4, 4))
        self.assertEqual(
            dataset[0],
            (
                [
                    [
                        [0.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0],
                    ],
                ],
                0.0,
            ),
        )

    def test_smoke_cnn_accepts_batched_channel_first_samples(self) -> None:
        dataset = smoke_dataset()
        model = SmokeCNN(seed=0)

        logits = model(Tensor.from_list(dataset.xs[:2]))

        self.assertEqual(logits.shape, (2, 2))

    def test_smoke_demo_runs(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            main()

        text = output.getvalue()
        self.assertIn("MNIST-style CNN smoke demo", text)
        self.assertIn("samples:      4", text)
        self.assertIn("input shape:  (1, 4, 4)", text)
        self.assertIn("accuracy:     1.000", text)


if __name__ == "__main__":
    unittest.main()
