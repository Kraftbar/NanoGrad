import unittest

from datasets import TinyDataset, make_batches, tiny_2d_clusters


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
        self.assertEqual(dataset[0], ([1.0, 2.0], 0.0))
        self.assertEqual(dataset[1], ([3.0, 4.0], 1.0))
        self.assertEqual(y, 0.0)

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
            list(make_batches([[1]], [0], batch_size=0))


def _flatten_batches(batches) -> list[tuple[tuple[float, ...], float]]:
    return [
        (tuple(x), y)
        for xs, ys in batches
        for x, y in zip(xs, ys)
    ]


if __name__ == "__main__":
    unittest.main()
