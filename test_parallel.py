import random
import unittest

from datasets import TinySequenceDataset
from language import CharTransformerModel
from parallel import DataParallel, _split, sequence_loss, target_count
from train import train_tensor_sequence_multiclass_dataset


def small_model(seed: int = 0) -> CharTransformerModel:
    return CharTransformerModel(7, context_size=5, embedding_dim=8, num_heads=2, num_layers=1, seed=seed)


def random_batch(rows: int, seed: int = 0):
    rng = random.Random(seed)
    xs = [[rng.randrange(7) for _ in range(5)] for _ in range(rows)]
    ys = [[rng.randrange(7) for _ in range(5)] for _ in range(rows)]
    return xs, ys


class DataParallelTests(unittest.TestCase):
    def test_split_is_contiguous_and_balanced(self) -> None:
        shards = _split(list(range(10)), list(range(10)), 4)

        self.assertEqual([xs for xs, _ in shards], [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9]])
        self.assertEqual(_split([1], [1], 3), [([1], [1]), ([], []), ([], [])])
        self.assertEqual(target_count([[1, 2, 3], [4, 5, 6]]), 6)
        self.assertEqual(target_count([1, 2]), 2)

    def test_gradients_match_a_single_process(self) -> None:
        model = small_model()
        xs, ys = random_batch(7)   # 7 rows over 3 workers: uneven slices
        loss = sequence_loss(model, xs, ys)
        model.zero_grad()
        loss.backward()
        expected = [parameter.grad[:] for parameter in model.parameters()]

        with DataParallel(model, sequence_loss, workers=3) as parallel:
            parallel_loss = parallel.loss_and_gradients(xs, ys)

        self.assertAlmostEqual(parallel_loss, loss.item(), places=12)
        for reference, parameter in zip(expected, model.parameters()):
            for a, b in zip(reference, parameter.grad):
                self.assertAlmostEqual(a, b, places=12)

    def test_workers_follow_weight_updates_and_small_batches(self) -> None:
        model = small_model()
        xs, ys = random_batch(2)

        with DataParallel(model, sequence_loss, workers=4) as parallel:   # more workers than rows
            first = parallel.loss_and_gradients(xs, ys)
            for parameter in model.parameters():
                parameter.data = [value * 0.5 for value in parameter.data]
            second = parallel.loss_and_gradients(xs, ys)

        self.assertAlmostEqual(second, sequence_loss(model, xs, ys).item(), places=12)
        self.assertNotAlmostEqual(first, second)

    def test_training_loop_gives_the_same_run_with_workers(self) -> None:
        xs, ys = random_batch(12, seed=1)
        histories = []
        for workers in (1, 3):
            model = small_model(seed=2)
            summary = train_tensor_sequence_multiclass_dataset(
                model,
                TinySequenceDataset(xs, ys),
                epochs=2,
                batch_size=6,
                lr=0.01,
                optimizer_name="adam",
                shuffle=False,
                workers=workers,
            )
            histories.append(summary.history)

        for single, parallel in zip(*histories):
            self.assertAlmostEqual(single, parallel, places=10)

    def test_workers_draw_different_dropout_masks(self) -> None:
        model = CharTransformerModel(7, context_size=5, embedding_dim=8, dropout=0.5, seed=0)
        xs, ys = random_batch(1)

        # The same row twice: with independent masks the two halves differ,
        # so the gradient is not just the one-worker gradient of that row.
        with DataParallel(model, sequence_loss, workers=2) as parallel:
            both = parallel.loss_and_gradients(xs * 2, ys * 2)
        with DataParallel(model, sequence_loss, workers=1) as parallel:
            one = parallel.loss_and_gradients(xs, ys)

        self.assertNotAlmostEqual(both, one)

    def test_rejects_zero_workers(self) -> None:
        with self.assertRaises(ValueError):
            DataParallel(small_model(), sequence_loss, workers=0)


if __name__ == "__main__":
    unittest.main()
