import contextlib
import io
import random
import tempfile
import unittest
from pathlib import Path

import recommend


def write_toy_data(folder: Path, users: int = 30, items: int = 40, seed: int = 0) -> None:
    """Users like items of their own parity, so the pattern is learnable."""

    rng = random.Random(seed)
    train, test, negatives = [], [], []
    for user in range(users):
        liked = [item for item in range(items) if item % 2 == user % 2]
        rng.shuffle(liked)
        held_out, rest = liked[0], liked[1:12]
        train += [f"{user}\t{item}\t5\t0" for item in rest]
        test.append(f"{user}\t{held_out}\t5\t0")
        unseen = [item for item in range(items) if item % 2 != user % 2]
        negatives.append("\t".join([f"({user},{held_out})"] + [str(item) for item in rng.sample(unseen, 15)]))
    (folder / "ml-1m.train.rating").write_text("\n".join(train) + "\n")
    (folder / "ml-1m.test.rating").write_text("\n".join(test) + "\n")
    (folder / "ml-1m.test.negative").write_text("\n".join(negatives) + "\n")


def quiet_main(argv):
    with contextlib.redirect_stdout(io.StringIO()):
        return recommend.main(argv)


class RecommendTest(unittest.TestCase):
    def test_ratings_and_negative_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            write_toy_data(Path(folder))
            ratings = recommend.Ratings(folder)
        self.assertEqual((ratings.users, ratings.items), (30, 40))
        self.assertEqual(len(ratings.test[0][1]), 16)
        examples = ratings.epoch(4, random.Random(0))
        self.assertEqual(len(examples), 5 * len(ratings.train))
        for user, item, label in examples:
            self.assertEqual(label == 1.0, item in ratings.seen[user])

    def test_neumf_learns_the_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            write_toy_data(Path(folder))
            result = quiet_main(["--data", folder, "--model", "neumf", "--epochs", "8", "--batch-size", "32",
                                 "--lr", "0.01", "--layers", "16", "8"])
        self.assertGreater(result["nanograd"]["hr"], 0.9)

    def test_matches_torch(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")
        for model in ("gmf", "mlp"):
            with tempfile.TemporaryDirectory() as folder:
                write_toy_data(Path(folder))
                result = quiet_main(["--data", folder, "--model", model, "--epochs", "1", "--batch-size", "16",
                                     "--frameworks", "nanograd", "torch", "--layers", "16", "8"])
            self.assertLess(result["max_weight_difference"], 1e-10)


if __name__ == "__main__":
    unittest.main()
