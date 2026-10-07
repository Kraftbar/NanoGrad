import math
import unittest

from char_demo import learning_rate_schedule, parse_args
from datasets import TinySequenceDataset
from language import CharTransformerModel
from schedules import constant, warmup_cosine
from train import train_tensor_sequence_multiclass_dataset


def nanogpt_get_lr(it, learning_rate, min_lr, warmup_iters, lr_decay_iters):
    """ref_nanogpt/train.py's get_lr, with its globals turned into arguments."""

    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


class ScheduleTests(unittest.TestCase):
    def test_warmup_cosine_matches_nanogpt(self) -> None:
        schedule = warmup_cosine(max_lr=1e-3, min_lr=1e-4, warmup_steps=100, decay_steps=5000)

        for step in (0, 1, 50, 99, 100, 101, 2500, 4999, 5000, 5001, 9000):
            self.assertAlmostEqual(
                schedule(step),
                nanogpt_get_lr(step, 1e-3, 1e-4, 100, 5000),
                places=15,
            )

    def test_shape(self) -> None:
        schedule = warmup_cosine(max_lr=1.0, min_lr=0.1, warmup_steps=10, decay_steps=110)
        values = [schedule(step) for step in range(120)]

        self.assertTrue(all(a < b for a, b in zip(values[:10], values[1:11])))   # warmup rises
        self.assertAlmostEqual(schedule(10), 1.0)                                  # peak
        self.assertAlmostEqual(schedule(60), 0.55)                                 # cosine midpoint
        self.assertTrue(all(a >= b for a, b in zip(values[10:110], values[11:111])))
        self.assertEqual(schedule(119), 0.1)
        self.assertEqual(constant(0.3)(12345), 0.3)

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            warmup_cosine(max_lr=0.1, min_lr=0.2, warmup_steps=0, decay_steps=10)
        with self.assertRaises(ValueError):
            warmup_cosine(max_lr=0.1, min_lr=0.0, warmup_steps=10, decay_steps=10)
        with self.assertRaises(ValueError):
            constant(0.0)

    def test_training_loop_follows_the_schedule(self) -> None:
        xs = [[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 2, 1]]
        ys = [[1, 2, 0], [2, 0, 1], [0, 1, 2], [2, 1, 0]]
        model = CharTransformerModel(3, context_size=3, embedding_dim=4, seed=0)
        seen = []

        def schedule(step: int) -> float:
            seen.append(step)
            return 0.01 / (step + 1)

        train_tensor_sequence_multiclass_dataset(
            model,
            TinySequenceDataset(xs, ys),
            epochs=2,
            batch_size=2,
            optimizer_name="adam",
            lr=0.01,
            lr_schedule=schedule,
        )

        self.assertEqual(seen, [0, 1, 2, 3])

    def test_char_demo_flags_build_a_run_long_schedule(self) -> None:
        args = parse_args(["--epochs", "2", "--batch-size", "4", "--lr", "0.01",
                           "--warmup-steps", "2", "--min-lr", "0.001"])
        dataset = TinySequenceDataset([[0]] * 10, [[0]] * 10)

        schedule = learning_rate_schedule(args, dataset)

        # 10 examples / batch 4 -> 3 steps per epoch -> 6 steps; decay ends at step 5.
        self.assertAlmostEqual(schedule(1), 0.01 * 2 / 3)
        self.assertAlmostEqual(schedule(5), 0.001)
        self.assertIsNone(learning_rate_schedule(parse_args([]), dataset))


if __name__ == "__main__":
    unittest.main()
