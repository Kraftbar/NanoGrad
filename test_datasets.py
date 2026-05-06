import gzip
import struct
import tempfile
import unittest
from pathlib import Path

from datasets import (
    TinyDataset,
    load_cifar10_batches,
    load_mnist,
    make_batches,
    read_cifar10_batch,
    read_cifar10_batches,
    read_mnist_images,
    read_mnist_labels,
    tiny_2d_clusters,
)


class DatasetTests(unittest.TestCase):
    def test_tiny_dataset_indexes_samples(self) -> None:
        dataset = TinyDataset(
            xs=[
                [1, 2],
                [3, 4],
            ],
            ys=[
                0,
                1,
            ],
        )

        x, y = dataset[0]
        x[0] = 99

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.feature_shape, (2,))
        self.assertEqual(dataset[0], ([1.0, 2.0], 0.0))
        self.assertEqual(dataset[1], ([3.0, 4.0], 1.0))
        self.assertEqual(y, 0.0)

    def test_tiny_dataset_preserves_nested_feature_samples(self) -> None:
        dataset = TinyDataset(
            xs=[
                [
                    [
                        [1, 2],
                        [3, 4],
                    ],
                ],
                [
                    [
                        [5, 6],
                        [7, 8],
                    ],
                ],
            ],
            ys=[
                0,
                1,
            ],
        )

        sample, _target = dataset[0]
        sample[0][0][0] = 99
        batch_xs, batch_ys = next(dataset.batches(batch_size=2))
        batch_xs[0][0][0][0] = 88

        self.assertEqual(dataset.feature_shape, (1, 2, 2))
        self.assertEqual(
            dataset[0],
            (
                [
                    [
                        [1.0, 2.0],
                        [3.0, 4.0],
                    ],
                ],
                0.0,
            ),
        )
        self.assertEqual(
            batch_ys,
            [0.0, 1.0],
        )

    def test_batches_preserve_order_and_keep_remainder(self) -> None:
        xs, ys = tiny_2d_clusters()
        batches = list(make_batches(xs, ys, batch_size=4))

        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0][0], xs[:4])
        self.assertEqual(batches[0][1], ys[:4])
        self.assertEqual(batches[1][0], xs[4:])
        self.assertEqual(batches[1][1], ys[4:])

    def test_batches_shuffle_deterministically(self) -> None:
        xs, ys = tiny_2d_clusters()
        ordered = list(make_batches(xs, ys, batch_size=2))
        shuffled_once = list(make_batches(xs, ys, batch_size=2, shuffle=True, seed=1))
        shuffled_twice = list(make_batches(xs, ys, batch_size=2, shuffle=True, seed=1))

        self.assertEqual(shuffled_once, shuffled_twice)
        self.assertNotEqual(shuffled_once, ordered)
        self.assertEqual(
            sorted(_flatten_batches(shuffled_once)),
            sorted((tuple(x), y) for x, y in zip(xs, ys)),
        )

    def test_dataset_validation_errors(self) -> None:
        with self.assertRaises(ValueError):
            TinyDataset([], [])

        with self.assertRaises(ValueError):
            TinyDataset([[1]], [0, 1])

        with self.assertRaises(ValueError):
            TinyDataset([[]], [0])

        with self.assertRaises(ValueError):
            TinyDataset([[1], [1, 2]], [0, 1])

        with self.assertRaises(ValueError):
            TinyDataset([[[1], [2]], [[3, 4]]], [0, 1])

        with self.assertRaises(ValueError):
            list(make_batches([[1]], [0], batch_size=0))

    def test_load_mnist_from_local_idx_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            images_path = Path(tmpdir) / "images.idx3-ubyte"
            labels_path = Path(tmpdir) / "labels.idx1-ubyte"
            _write_mnist_images(images_path)
            _write_mnist_labels(labels_path)

            dataset = load_mnist(images_path, labels_path)

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0], ([0.0, 1.0 / 255.0, 2.0 / 255.0, 1.0], 7.0))
        self.assertEqual(dataset[1], ([1.0, 0.0, 128.0 / 255.0, 64.0 / 255.0], 3.0))

    def test_load_mnist_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            images_path = Path(tmpdir) / "images.idx3-ubyte"
            labels_path = Path(tmpdir) / "labels.idx1-ubyte"
            _write_mnist_images(images_path)
            _write_mnist_labels(labels_path)

            dataset = load_mnist(images_path, labels_path, limit=1)

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0][1], 7.0)

    def test_load_mnist_channel_first_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            images_path = Path(tmpdir) / "images.idx3-ubyte"
            labels_path = Path(tmpdir) / "labels.idx1-ubyte"
            _write_mnist_images(images_path)
            _write_mnist_labels(labels_path)

            dataset = load_mnist(images_path, labels_path, channel_first=True)

        self.assertEqual(dataset.feature_shape, (1, 2, 2))
        self.assertEqual(
            dataset[0],
            (
                [
                    [
                        [0.0, 1.0 / 255.0],
                        [2.0 / 255.0, 1.0],
                    ],
                ],
                7.0,
            ),
        )

    def test_load_mnist_from_gzip_idx_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            images_path = Path(tmpdir) / "images.idx3-ubyte.gz"
            labels_path = Path(tmpdir) / "labels.idx1-ubyte.gz"
            _write_mnist_images(images_path, gzip_output=True)
            _write_mnist_labels(labels_path, gzip_output=True)

            dataset = load_mnist(images_path, labels_path)

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[1][1], 3.0)

    def test_mnist_validation_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            images_path = Path(tmpdir) / "bad-images.idx3-ubyte"
            labels_path = Path(tmpdir) / "bad-labels.idx1-ubyte"

            images_path.write_bytes(struct.pack(">IIII", 9999, 1, 2, 2) + b"\x00" * 4)
            labels_path.write_bytes(struct.pack(">II", 9999, 1) + b"\x00")

            with self.assertRaises(ValueError):
                read_mnist_images(images_path)

            with self.assertRaises(ValueError):
                read_mnist_labels(labels_path)

            with self.assertRaises(ValueError):
                read_mnist_images(images_path, limit=-1)

    def test_mnist_truncation_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            images_path = Path(tmpdir) / "images.idx3-ubyte"
            labels_path = Path(tmpdir) / "labels.idx1-ubyte"

            images_path.write_bytes(struct.pack(">IIII", 2051, 1, 2, 2) + b"\x00")
            labels_path.write_bytes(struct.pack(">II", 2049, 2) + b"\x00")

            with self.assertRaises(ValueError):
                read_mnist_images(images_path)

            with self.assertRaises(ValueError):
                read_mnist_labels(labels_path)

    def test_load_cifar10_from_local_binary_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_path = Path(tmpdir) / "data_batch_1.bin"
            _write_cifar10_batch(batch_path, labels=[3, 7])

            dataset = load_cifar10_batches([batch_path])

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.feature_shape, (3, 32, 32))
        self.assertEqual(dataset[0][1], 3.0)
        self.assertEqual(dataset[1][1], 7.0)
        self.assertEqual(dataset[0][0][0][0][0], 0.0)
        self.assertEqual(dataset[0][0][1][0][0], 1.0)
        self.assertEqual(dataset[0][0][2][0][0], 128.0 / 255.0)

    def test_read_cifar10_flat_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_path = Path(tmpdir) / "data_batch_1.bin"
            _write_cifar10_batch(batch_path, labels=[4])

            images, labels = read_cifar10_batch(batch_path, channel_first=False)

        self.assertEqual(labels, [4.0])
        self.assertEqual(len(images[0]), 3 * 32 * 32)
        self.assertEqual(images[0][0], 0.0)
        self.assertEqual(images[0][1024], 1.0)
        self.assertEqual(images[0][2048], 128.0 / 255.0)

    def test_read_cifar10_multiple_batches_and_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            first_path = Path(tmpdir) / "data_batch_1.bin"
            second_path = Path(tmpdir) / "data_batch_2.bin"
            _write_cifar10_batch(first_path, labels=[1, 2])
            _write_cifar10_batch(second_path, labels=[3, 4])

            images, labels = read_cifar10_batches(
                [first_path, second_path],
                limit=3,
            )

        self.assertEqual(len(images), 3)
        self.assertEqual(labels, [1.0, 2.0, 3.0])

    def test_cifar10_validation_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_path = Path(tmpdir) / "bad_batch.bin"
            batch_path.write_bytes(b"\x00")

            with self.assertRaises(ValueError):
                read_cifar10_batch(batch_path)

            with self.assertRaises(ValueError):
                read_cifar10_batch(batch_path, limit=-1)

            with self.assertRaises(ValueError):
                read_cifar10_batches([])


def _flatten_batches(batches) -> list[tuple[tuple[float, ...], float]]:
    return [
        (tuple(x), y)
        for xs, ys in batches
        for x, y in zip(xs, ys)
    ]


def _write_mnist_images(path: Path, *, gzip_output: bool = False) -> None:
    payload = struct.pack(">IIII", 2051, 2, 2, 2) + bytes([
        0,
        1,
        2,
        255,
        255,
        0,
        128,
        64,
    ])
    _write_bytes(path, payload, gzip_output=gzip_output)


def _write_mnist_labels(path: Path, *, gzip_output: bool = False) -> None:
    payload = struct.pack(">II", 2049, 2) + bytes([7, 3])
    _write_bytes(path, payload, gzip_output=gzip_output)


def _write_cifar10_batch(path: Path, *, labels: list[int]) -> None:
    payload = b"".join(
        bytes([label])
        + bytes([0] * (32 * 32))
        + bytes([255] * (32 * 32))
        + bytes([128] * (32 * 32))
        for label in labels
    )
    path.write_bytes(payload)


def _write_bytes(path: Path, payload: bytes, *, gzip_output: bool) -> None:
    if gzip_output:
        with gzip.open(path, "wb") as file:
            file.write(payload)
        return
    path.write_bytes(payload)


if __name__ == "__main__":
    unittest.main()
