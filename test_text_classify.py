import contextlib
import csv
import importlib.util
import io
import random
import tempfile
import unittest
from pathlib import Path

from text_classify import BagOfEmbeddings, Features, main, tokenize


HAS_TORCH = importlib.util.find_spec("torch") is not None
TOPICS = (["match", "goal", "team", "win"], ["market", "stock", "profit", "bank"],
          ["vote", "minister", "war", "election"], ["software", "chip", "web", "space"])


def tiny_news(folder: Path) -> None:
    rng = random.Random(0)
    for name, count in (("train", 400), ("test", 80)):
        with (folder / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for _ in range(count):
                label = rng.randrange(4)
                words = [rng.choice(TOPICS[label]) for _ in range(4)] + [rng.choice(TOPICS[rng.randrange(4)])]
                writer.writerow([label + 1, " ".join(words[:2]), " ".join(words[2:]) + "."])


class TextClassifyTests(unittest.TestCase):
    def test_tokenize_splits_words_and_punctuation(self) -> None:
        self.assertEqual(tokenize("Wall St. Bears (Reuters)"), ["wall", "st", ".", "bears", "(", "reuters", ")"])

    def test_bag_is_the_mean_of_its_embeddings(self) -> None:
        model = BagOfEmbeddings(6, 4, dim=3, seed=1)
        model.output.weight.data = [1.0 if row == col else 0.0 for row in range(4) for col in range(3)]
        scores = model([[1, 2], [5]]).data
        table = model.embedding.data
        for column in range(3):
            self.assertAlmostEqual(scores[column], (table[3 + column] + table[6 + column]) / 2)
            self.assertAlmostEqual(scores[4 + column], table[15 + column])

    def test_bigrams_add_hashed_ids(self) -> None:
        features = Features([(["a", "b", "c"], 0)], bigrams=True, buckets=7)
        ids = features.ids(["a", "b", "c"])
        self.assertEqual(len(ids), 3 + 2)
        self.assertTrue(all(3 <= index < 10 for index in ids[3:]))

    def test_learns_topics_and_matches_pytorch(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tiny_news(Path(folder))
            frameworks = ["nanograd", "torch"] if HAS_TORCH else ["nanograd"]
            with contextlib.redirect_stdout(io.StringIO()):
                results = main(["--data", folder, "--epochs", "3", "--frameworks", *frameworks])
        self.assertGreater(results["nanograd"]["accuracy"], 0.9)
        if HAS_TORCH:
            self.assertLess(results["max_weight_difference"], 1e-12)


if __name__ == "__main__":
    unittest.main()
