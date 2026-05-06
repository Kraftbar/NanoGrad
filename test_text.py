import unittest

from text import CharVocab, next_char_dataset, next_char_index_dataset


class TextHelperTests(unittest.TestCase):
    def test_char_vocab_encodes_and_decodes_text(self) -> None:
        vocab = CharVocab.from_text("cab")

        self.assertEqual(vocab.itos, ["a", "b", "c"])
        self.assertEqual(vocab.encode("abc"), [0, 1, 2])
        self.assertEqual(vocab.decode([2, 0, 1]), "cab")
        self.assertEqual(vocab.one_hot(1), [0.0, 1.0, 0.0])

    def test_next_char_dataset_uses_one_hot_contexts(self) -> None:
        dataset, vocab = next_char_dataset("abca")

        self.assertEqual(vocab.itos, ["a", "b", "c"])
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.feature_shape, (3,))
        self.assertEqual(dataset[0], ([1.0, 0.0, 0.0], 1.0))
        self.assertEqual(dataset[1], ([0.0, 1.0, 0.0], 2.0))
        self.assertEqual(dataset[2], ([0.0, 0.0, 1.0], 0.0))

    def test_next_char_dataset_supports_larger_contexts(self) -> None:
        dataset, vocab = next_char_dataset("abca", context_size=2)

        self.assertEqual(len(vocab), 3)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.feature_shape, (6,))
        self.assertEqual(
            dataset[0],
            (
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ],
                2.0,
            ),
        )

    def test_next_char_index_dataset_uses_token_id_contexts(self) -> None:
        dataset, vocab = next_char_index_dataset("abca", context_size=2)

        self.assertEqual(vocab.itos, ["a", "b", "c"])
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.feature_shape, (2,))
        self.assertEqual(dataset[0], ([0.0, 1.0], 2.0))
        self.assertEqual(dataset[1], ([1.0, 2.0], 0.0))

    def test_next_char_dataset_can_reuse_vocab(self) -> None:
        vocab = CharVocab.from_text("abcd")
        dataset, returned_vocab = next_char_dataset("cda", vocab=vocab)

        self.assertIs(returned_vocab, vocab)
        self.assertEqual(dataset[0], ([0.0, 0.0, 1.0, 0.0], 3.0))
        self.assertEqual(dataset[1], ([0.0, 0.0, 0.0, 1.0], 0.0))

    def test_char_vocab_errors(self) -> None:
        with self.assertRaises(ValueError):
            CharVocab("")

        with self.assertRaises(ValueError):
            CharVocab(["ab"])

        vocab = CharVocab.from_text("ab")
        with self.assertRaises(ValueError):
            vocab.encode("abc")

        with self.assertRaises(ValueError):
            vocab.decode([2])

        with self.assertRaises(ValueError):
            vocab.one_hot(-1)

    def test_next_char_dataset_errors(self) -> None:
        with self.assertRaises(ValueError):
            next_char_dataset("ab", context_size=0)

        with self.assertRaises(ValueError):
            next_char_dataset("ab", context_size=2)

        with self.assertRaises(ValueError):
            next_char_index_dataset("ab", context_size=0)

        with self.assertRaises(ValueError):
            next_char_index_dataset("ab", context_size=2)


if __name__ == "__main__":
    unittest.main()
