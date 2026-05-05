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
        ])

        self.assertEqual(args.data_dir, Path("custom"))
        self.assertEqual(args.limit, 4)
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.batch_size, 2)
        self.assertEqual(args.lr, 0.1)
        self.assertEqual(args.hidden, 5)
        self.assertEqual(args.seed, 9)

    def test_run_trains_on_tiny_idx_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
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
                "--lr",
                "0.1",
            ])

            output = io.StringIO()
            with redirect_stdout(output):
                run(args)

        text = output.getvalue()
        self.assertIn("MNIST MLP demo", text)
        self.assertIn("samples:      4", text)
        self.assertIn("inputs:       4", text)
        self.assertIn("classes:      2", text)


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
