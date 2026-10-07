import tempfile
import unittest
from pathlib import Path

from bpe import SPLIT_PATTERN, BPETokenizer, _merge


TEXT = (
    "First Citizen:\nBefore we proceed any further, hear me speak.\n\n"
    "All:\nSpeak, speak.\n\nFirst Citizen:\nYou are all resolved rather to die than to famish?\n"
) * 5


class BPETests(unittest.TestCase):
    def test_merge_replaces_non_overlapping_pairs_left_to_right(self) -> None:
        self.assertEqual(_merge([1, 1, 1, 2, 1, 1], (1, 1), 9), [9, 1, 2, 9])

    def test_split_pattern_covers_every_character(self) -> None:
        for text in (TEXT, "  two  spaces\tand tabs\n\n", "it's 42, naïve café!"):
            self.assertEqual("".join(SPLIT_PATTERN.findall(text)), text)

    def test_training_learns_frequent_pairs_and_round_trips(self) -> None:
        tokenizer = BPETokenizer.train(TEXT, 300)
        self.assertEqual(len(tokenizer), 300)
        tokens = tokenizer.encode(TEXT)
        self.assertLess(len(tokens), len(TEXT.encode("utf-8")) / 2)
        self.assertEqual(tokenizer.decode(tokens), TEXT)
        self.assertIn(" speak", {tokenizer.token_text(t) for t in range(256, 300)})

    def test_unseen_and_non_ascii_text_still_round_trips(self) -> None:
        tokenizer = BPETokenizer.train(TEXT, 280)
        for text in ("Zebras quietly juxtapose.", "naïve café ☕", ""):
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_save_and_load(self) -> None:
        tokenizer = BPETokenizer.train(TEXT, 290)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            loaded = BPETokenizer.load(path)
        self.assertEqual(loaded.merges, tokenizer.merges)
        self.assertEqual(loaded.encode(TEXT), tokenizer.encode(TEXT))

    def test_vocab_size_below_bytes_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BPETokenizer.train(TEXT, 100)


if __name__ == "__main__":
    unittest.main()
