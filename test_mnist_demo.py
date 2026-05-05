import gzip
import io
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from mnist_demo import find_mnist_files, parse_args, run


class MNISTDemoTests(unittest.TestCase):
    def test_find_mnist_files_prefers_gzip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            gz_images = data_dir / "train-images-idx3-ubyte.gz"
            gz_labels = data_dir / "train-labels-idx1-ubyte.gz"
            plain_images = data_dir / "train-images-idx3-ubyte"
            plain_labels = data_dir / "train-labels-idx1-ubyte"
            for path in (gz_images, gz_labels, plain_images, plain_labels):
                path.write_bytes(b"")

            self.assertEqual(find_mnist_files(data_dir), (gz_images, gz_labels))

    def test_find_mnist_files_accepts_plain_idx(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            images = data_dir / "train-images-idx3-ubyte"
            labels = data_dir / "train-labels-idx1-ubyte"
            images.write_bytes(b"")
            labels.write_bytes(b"")

            self.assertEqual(find_mnist_files(data_dir), (images, labels))

    def test_find_mnist_files_accepts_optional_test_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            images = data_dir / "t10k-images-idx3-ubyte"
            labels = data_dir / "t10k-labels-idx1-ubyte"
            images.write_bytes(b"")
            labels.write_bytes(b"")

            self.assertEqual(
                find_mnist_files(data_dir, split="test", required=False),
                (images, labels),
            )

    def test_find_mnist_files_optional_missing_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertIsNone(
                find_mnist_files(Path(tmpdir), split="test", required=False)
            )

    def test_find_mnist_files_split_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                find_mnist_files(Path(tmpdir), split="validation")

    def test_find_mnist_files_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                find_mnist_files(Path(tmpdir))

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
            "--hidden",
            "5",
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
        self.assertEqual(args.hidden, 5)
        self.assertEqual(args.seed, 9)
        self.assertEqual(args.report_every, 2)
        self.assertEqual(args.save_model, Path("model.json"))
        self.assertEqual(args.load_model, Path("model-in.json"))
        self.assertTrue(args.check_data)

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
                "--hidden",
                "3",
                "--lr",
                "0.1",
                "--report-every",
                "1",
            ])

            output = io.StringIO()
            with redirect_stdout(output):
                run(args)

        text = output.getvalue()
        self.assertIn("MNIST MLP demo", text)
        self.assertIn("samples:      4", text)
        self.assertIn("inputs:       4", text)
        self.assertIn("classes:      2", text)
        self.assertIn("val accuracy:", text)
        self.assertIn("epoch 1/2", text)
        self.assertIn("val_acc=", text)

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
        self.assertIn("MNIST data check", text)
        self.assertIn("train samples: 2", text)
        self.assertIn("train inputs:  4", text)
        self.assertIn("train labels:  [0]", text)
        self.assertIn("test samples: 1", text)

    def test_run_saves_and_loads_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            save_path = data_dir / "mnist-model.json"
            reload_path = data_dir / "mnist-model-reloaded.json"
            _write_mnist_images(data_dir / "train-images-idx3-ubyte.gz")
            _write_mnist_labels(data_dir / "train-labels-idx1-ubyte.gz")
            args = parse_args([
                "--data-dir",
                str(data_dir),
                "--limit",
                "4",
                "--epochs",
                "2",
                "--batch-size",
                "2",
                "--hidden",
                "3",
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
                "--hidden",
                "3",
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
