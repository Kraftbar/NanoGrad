import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from cifar_cnn_demo import (
    build_model,
    find_cifar10_files,
    find_cifar10_test_file,
    parse_args,
    run,
)
from vision import SimpleCNN, TwoConvCNN


class CIFARCnnDemoTests(unittest.TestCase):
    def test_parse_args(self) -> None:
        args = parse_args([
            "--data-dir",
            "custom",
            "--limit",
            "4",
            "--validation-limit",
            "3",
            "--epochs",
            "2",
            "--batch-size",
            "1",
            "--lr",
            "0.1",
            "--normalize",
            "train",
            "--architecture",
            "two-conv",
            "--activation",
            "tanh",
            "--filters",
            "2",
            "--second-filters",
            "3",
            "--kernel-size",
            "5",
            "--pool-size",
            "2",
            "--seed",
            "9",
            "--report-every",
            "1",
            "--save-model",
            "model.json",
            "--load-model",
            "model-in.json",
            "--show-confusion",
            "--check-data",
        ])

        self.assertEqual(args.data_dir, Path("custom"))
        self.assertEqual(args.limit, 4)
        self.assertEqual(args.validation_limit, 3)
        self.assertEqual(args.epochs, 2)
        self.assertEqual(args.batch_size, 1)
        self.assertEqual(args.lr, 0.1)
        self.assertEqual(args.normalize, "train")
        self.assertEqual(args.architecture, "two-conv")
        self.assertEqual(args.activation, "tanh")
        self.assertEqual(args.filters, 2)
        self.assertEqual(args.second_filters, 3)
        self.assertEqual(args.kernel_size, 5)
        self.assertEqual(args.pool_size, 2)
        self.assertEqual(args.seed, 9)
        self.assertEqual(args.report_every, 1)
        self.assertEqual(args.save_model, Path("model.json"))
        self.assertEqual(args.load_model, Path("model-in.json"))
        self.assertTrue(args.show_confusion)
        self.assertTrue(args.check_data)

    def test_build_model_uses_requested_architecture(self) -> None:
        simple_args = parse_args([])
        two_conv_args = parse_args(["--architecture", "two-conv"])

        self.assertIsInstance(
            build_model(simple_args, image_shape=(3, 32, 32), classes=10),
            SimpleCNN,
        )
        self.assertIsInstance(
            build_model(two_conv_args, image_shape=(3, 32, 32), classes=10),
            TwoConvCNN,
        )

    def test_find_cifar10_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            first = data_dir / "data_batch_1.bin"
            third = data_dir / "data_batch_3.bin"
            first.write_bytes(b"")
            third.write_bytes(b"")

            self.assertEqual(find_cifar10_files(data_dir), [first, third])
            self.assertIsNone(find_cifar10_test_file(data_dir, required=False))

    def test_find_cifar10_files_accepts_extracted_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            extracted_dir = data_dir / "cifar-10-batches-bin"
            extracted_dir.mkdir()
            train_path = extracted_dir / "data_batch_1.bin"
            test_path = extracted_dir / "test_batch.bin"
            train_path.write_bytes(b"")
            test_path.write_bytes(b"")

            self.assertEqual(find_cifar10_files(data_dir), [train_path])
            self.assertEqual(find_cifar10_test_file(data_dir), test_path)

    def test_find_cifar10_files_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                find_cifar10_files(Path(tmpdir))

            with self.assertRaises(FileNotFoundError):
                find_cifar10_test_file(Path(tmpdir))

    def test_run_check_data_on_tiny_binary_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            _write_cifar10_batch(data_dir / "data_batch_1.bin", labels=[0, 1])
            _write_cifar10_batch(data_dir / "test_batch.bin", labels=[1])
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
        self.assertIn("CIFAR-10 data check", text)
        self.assertIn("train batches: 1", text)
        self.assertIn("train samples: 2", text)
        self.assertIn("train shape:   (3, 32, 32)", text)
        self.assertIn("test samples: 1", text)

    def test_run_check_data_with_train_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            _write_cifar10_batch(data_dir / "data_batch_1.bin", labels=[0, 1])
            args = parse_args([
                "--data-dir",
                str(data_dir),
                "--limit",
                "2",
                "--normalize",
                "train",
                "--check-data",
            ])

            output = io.StringIO()
            with redirect_stdout(output):
                run(args)

        text = output.getvalue()
        self.assertIn("normalize:     train", text)
        self.assertIn("channel mean:", text)
        self.assertIn("channel std:", text)
        self.assertIn("test:          not found", text)

    def test_run_trains_on_tiny_binary_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            _write_cifar10_batch(data_dir / "data_batch_1.bin", labels=[0, 1])
            _write_cifar10_batch(data_dir / "test_batch.bin", labels=[1, 0])
            args = parse_args([
                "--data-dir",
                str(data_dir),
                "--limit",
                "2",
                "--validation-limit",
                "2",
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--normalize",
                "train",
                "--filters",
                "1",
                "--kernel-size",
                "32",
                "--pool-size",
                "1",
                "--show-confusion",
            ])

            output = io.StringIO()
            with redirect_stdout(output):
                run(args)

        text = output.getvalue()
        self.assertIn("CIFAR-10 CNN demo", text)
        self.assertIn("samples:      2", text)
        self.assertIn("input shape:  (3, 32, 32)", text)
        self.assertIn("classes:      10", text)
        self.assertIn("normalize:    train", text)
        self.assertIn("channel mean:", text)
        self.assertIn("parameters:", text)
        self.assertIn("train loss:", text)
        self.assertIn("val loss:", text)
        self.assertIn("train confusion:", text)
        self.assertIn("val confusion:", text)

    def test_run_trains_with_two_conv_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            _write_cifar10_batch(data_dir / "data_batch_1.bin", labels=[0])
            _write_cifar10_batch(data_dir / "test_batch.bin", labels=[0])
            args = parse_args([
                "--data-dir",
                str(data_dir),
                "--limit",
                "1",
                "--validation-limit",
                "1",
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--architecture",
                "two-conv",
                "--filters",
                "1",
                "--second-filters",
                "1",
                "--kernel-size",
                "3",
                "--pool-size",
                "2",
            ])

            output = io.StringIO()
            with redirect_stdout(output):
                run(args)

        text = output.getvalue()
        self.assertIn("architecture: two-conv", text)
        self.assertIn("filters 2:    1", text)
        self.assertIn("val loss:", text)


def _write_cifar10_batch(path: Path, *, labels: list[int]) -> None:
    payload = b"".join(
        bytes([label])
        + bytes([label] * (32 * 32))
        + bytes([255 - label] * (32 * 32))
        + bytes([128] * (32 * 32))
        for label in labels
    )
    path.write_bytes(payload)


if __name__ == "__main__":
    unittest.main()
