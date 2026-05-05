import gzip
import io
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from mnist_cnn_demo import (
    LeNetishCNN,
    MNISTCNN,
    _activate,
    _apply_preset,
    build_model,
    parse_args,
    run,
)
from tensor import Tensor


class MNISTCNNDemoTests(unittest.TestCase):
    def test_parse_args(self) -> None:
        args = parse_args([
            "--data-dir",
            "custom",
            "--limit",
            "4",
            "--validation-limit",
            "3",
            "--epochs",
            "3",
            "--batch-size",
            "2",
            "--lr",
            "0.1",
            "--preset",
            "lenet-ish",
            "--architecture",
            "simple",
            "--activation",
            "relu",
            "--filters",
            "5",
            "--second-filters",
            "7",
            "--kernel-size",
            "2",
            "--pool-size",
            "1",
            "--seed",
            "9",
            "--report-every",
            "2",
            "--save-model",
            "model.json",
            "--load-model",
            "model-in.json",
            "--check-data",
        ])

        self.assertEqual(args.data_dir, Path("custom"))
        self.assertEqual(args.limit, 4)
        self.assertEqual(args.validation_limit, 3)
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.batch_size, 2)
        self.assertEqual(args.lr, 0.1)
        self.assertEqual(args.preset, "lenet-ish")
        self.assertEqual(args.architecture, "simple")
        self.assertEqual(args.activation, "relu")
        self.assertEqual(args.filters, 5)
        self.assertEqual(args.second_filters, 7)
        self.assertEqual(args.kernel_size, 2)
        self.assertEqual(args.pool_size, 1)
        self.assertEqual(args.seed, 9)
        self.assertEqual(args.report_every, 2)
        self.assertEqual(args.save_model, Path("model.json"))
        self.assertEqual(args.load_model, Path("model-in.json"))
        self.assertTrue(args.check_data)

    def test_cnn_preset_supplies_default_shape_values(self) -> None:
        args = _apply_preset(parse_args(["--preset", "lenet-ish"]))

        self.assertEqual(args.architecture, "lenet-ish")
        self.assertEqual(args.activation, "tanh")
        self.assertEqual(args.filters, 6)
        self.assertEqual(args.second_filters, 16)
        self.assertEqual(args.kernel_size, 5)
        self.assertEqual(args.pool_size, 2)

    def test_explicit_shape_values_override_preset(self) -> None:
        args = _apply_preset(parse_args([
            "--preset",
            "lenet-ish",
            "--architecture",
            "simple",
            "--activation",
            "relu",
            "--filters",
            "2",
            "--second-filters",
            "3",
            "--kernel-size",
            "3",
            "--pool-size",
            "1",
        ]))

        self.assertEqual(args.architecture, "simple")
        self.assertEqual(args.activation, "relu")
        self.assertEqual(args.filters, 2)
        self.assertEqual(args.second_filters, 3)
        self.assertEqual(args.kernel_size, 3)
        self.assertEqual(args.pool_size, 1)

    def test_mnist_cnn_forward_shape(self) -> None:
        model = MNISTCNN(
            image_shape=(1, 2, 2),
            classes=2,
            filters=3,
            kernel_size=2,
            pool_size=1,
            seed=0,
        )

        logits = model(Tensor.from_list([
            [
                [
                    [1.0, 0.0],
                    [0.0, 0.0],
                ],
            ],
            [
                [
                    [0.0, 0.0],
                    [0.0, 1.0],
                ],
            ],
        ]))

        self.assertEqual(logits.shape, (2, 2))

    def test_mnist_cnn_accepts_tanh_activation(self) -> None:
        model = MNISTCNN(
            image_shape=(1, 2, 2),
            classes=2,
            filters=3,
            kernel_size=2,
            pool_size=1,
            activation="tanh",
            seed=0,
        )

        logits = model(Tensor.from_list([
            [
                [
                    [1.0, 0.0],
                    [0.0, 0.0],
                ],
            ],
        ]))

        self.assertEqual(logits.shape, (1, 2))

    def test_lenetish_cnn_forward_shape(self) -> None:
        model = LeNetishCNN(
            image_shape=(1, 28, 28),
            classes=10,
            filters=6,
            second_filters=16,
            kernel_size=5,
            pool_size=2,
            activation="tanh",
            seed=0,
        )

        logits = model(Tensor.zeros((2, 1, 28, 28)))

        self.assertEqual(logits.shape, (2, 10))
        self.assertEqual(len(model.parameters()), 6)

    def test_build_model_uses_requested_architecture(self) -> None:
        simple_args = _apply_preset(parse_args(["--preset", "tiny"]))
        lenet_args = _apply_preset(parse_args(["--preset", "lenet-ish"]))

        self.assertIsInstance(
            build_model(simple_args, image_shape=(1, 4, 4), classes=2),
            MNISTCNN,
        )
        self.assertIsInstance(
            build_model(lenet_args, image_shape=(1, 28, 28), classes=10),
            LeNetishCNN,
        )

    def test_mnist_cnn_shape_errors(self) -> None:
        with self.assertRaises(ValueError):
            MNISTCNN(image_shape=(2, 2), classes=2)
        with self.assertRaises(ValueError):
            MNISTCNN(image_shape=(1, 2, 2), classes=0)
        with self.assertRaises(ValueError):
            MNISTCNN(image_shape=(1, 2, 2), classes=2, filters=0)
        with self.assertRaises(ValueError):
            MNISTCNN(image_shape=(1, 2, 2), classes=2, kernel_size=3)
        with self.assertRaises(ValueError):
            MNISTCNN(
                image_shape=(1, 2, 2),
                classes=2,
                kernel_size=2,
                pool_size=2,
            )
        with self.assertRaises(ValueError):
            MNISTCNN(
                image_shape=(1, 2, 2),
                classes=2,
                kernel_size=2,
                activation="sigmoid",
            )

        with self.assertRaises(ValueError):
            _activate(Tensor.from_list([1.0]), "sigmoid")

        with self.assertRaises(ValueError):
            LeNetishCNN(image_shape=(1, 12, 12), classes=10)

        args = _apply_preset(parse_args(["--preset", "tiny"]))
        args.architecture = "unknown"
        with self.assertRaises(ValueError):
            build_model(args, image_shape=(1, 4, 4), classes=2)

    def test_run_trains_on_tiny_idx_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            _write_mnist_images(data_dir / "train-images-idx3-ubyte.gz")
            _write_mnist_labels(data_dir / "train-labels-idx1-ubyte.gz")
            _write_mnist_images(data_dir / "t10k-images-idx3-ubyte.gz")
            _write_mnist_labels(data_dir / "t10k-labels-idx1-ubyte.gz")
            args = parse_args([
                "--data-dir",
                str(data_dir),
                "--limit",
                "4",
                "--epochs",
                "2",
                "--batch-size",
                "2",
                "--filters",
                "2",
                "--kernel-size",
                "2",
                "--pool-size",
                "1",
                "--lr",
                "0.1",
                "--report-every",
                "1",
            ])

            output = io.StringIO()
            with redirect_stdout(output):
                run(args)

        text = output.getvalue()
        self.assertIn("MNIST CNN demo", text)
        self.assertIn("samples:      4", text)
        self.assertIn("input shape:  (1, 2, 2)", text)
        self.assertIn("classes:      2", text)
        self.assertIn("preset:       tiny", text)
        self.assertIn("architecture: simple", text)
        self.assertIn("activation:   relu", text)
        self.assertIn("filters:      2", text)
        self.assertIn("final batch:", text)
        self.assertIn("train loss:", text)
        self.assertIn("val loss:", text)
        self.assertIn("val accuracy:", text)
        self.assertIn("epoch 1/2", text)

    def test_run_check_data_on_tiny_idx_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            _write_mnist_images(data_dir / "train-images-idx3-ubyte.gz")
            _write_mnist_labels(data_dir / "train-labels-idx1-ubyte.gz")
            _write_mnist_images(data_dir / "t10k-images-idx3-ubyte.gz")
            _write_mnist_labels(data_dir / "t10k-labels-idx1-ubyte.gz")
            args = parse_args([
                "--data-dir",
                str(data_dir),
                "--limit",
                "2",
                "--validation-limit",
                "1",
                "--check-data",
            ])

            output = io.StringIO()
            with redirect_stdout(output):
                run(args)

        text = output.getvalue()
        self.assertIn("MNIST CNN data check", text)
        self.assertIn("train samples: 2", text)
        self.assertIn("train shape:   (1, 2, 2)", text)
        self.assertIn("train labels:  [0]", text)
        self.assertIn("test samples: 1", text)

    def test_run_check_data_without_test_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            _write_mnist_images(data_dir / "train-images-idx3-ubyte.gz")
            _write_mnist_labels(data_dir / "train-labels-idx1-ubyte.gz")
            args = parse_args([
                "--data-dir",
                str(data_dir),
                "--limit",
                "1",
                "--check-data",
            ])

            output = io.StringIO()
            with redirect_stdout(output):
                run(args)

        self.assertIn("test:         not found", output.getvalue())

    def test_run_saves_and_loads_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            save_path = data_dir / "mnist-cnn.json"
            reload_path = data_dir / "mnist-cnn-reloaded.json"
            _write_mnist_images(data_dir / "train-images-idx3-ubyte.gz")
            _write_mnist_labels(data_dir / "train-labels-idx1-ubyte.gz")
            args = parse_args([
                "--data-dir",
                str(data_dir),
                "--limit",
                "4",
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--filters",
                "2",
                "--kernel-size",
                "2",
                "--pool-size",
                "1",
                "--save-model",
                str(save_path),
            ])
            with redirect_stdout(io.StringIO()):
                run(args)

            reload_args = parse_args([
                "--data-dir",
                str(data_dir),
                "--limit",
                "4",
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--filters",
                "2",
                "--kernel-size",
                "2",
                "--pool-size",
                "1",
                "--load-model",
                str(save_path),
                "--save-model",
                str(reload_path),
            ])
            output = io.StringIO()
            with redirect_stdout(output):
                run(reload_args)

            self.assertTrue(save_path.exists())
            self.assertTrue(reload_path.exists())
            self.assertIn("saved model:", output.getvalue())


def _write_mnist_images(path: Path) -> None:
    payload = struct.pack(">IIII", 2051, 4, 2, 2) + bytes([
        255,
        0,
        0,
        0,
        230,
        20,
        0,
        0,
        0,
        0,
        0,
        255,
        0,
        0,
        20,
        230,
    ])
    with gzip.open(path, "wb") as file:
        file.write(payload)


def _write_mnist_labels(path: Path) -> None:
    payload = struct.pack(">II", 2049, 4) + bytes([0, 0, 1, 1])
    with gzip.open(path, "wb") as file:
        file.write(payload)


if __name__ == "__main__":
    unittest.main()
