import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from char_demo import CharBigramModel, _argmax, generate_text, parse_args, run
from tensor import Tensor
from text import CharVocab


class CharDemoTests(unittest.TestCase):
    def test_parse_args(self) -> None:
        args = parse_args([
            "--text",
            "abba",
            "--epochs",
            "3",
            "--batch-size",
            "2",
            "--lr",
            "0.1",
            "--seed",
            "9",
            "--seed-text",
            "a",
            "--generate",
            "5",
        ])

        self.assertEqual(args.text, "abba")
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.batch_size, 2)
        self.assertEqual(args.lr, 0.1)
        self.assertEqual(args.seed, 9)
        self.assertEqual(args.seed_text, "a")
        self.assertEqual(args.generate, 5)

    def test_char_bigram_model_forward_shape(self) -> None:
        model = CharBigramModel(3, seed=0)

        logits = model(Tensor.from_list([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]))

        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(model.num_parameters(), 12)

    def test_generate_text_uses_argmax_predictions(self) -> None:
        vocab = CharVocab.from_text("ab")
        model = CharBigramModel(len(vocab), seed=0)
        model.projection.weight.data = [
            0.0,
            1.0,
            2.0,
            0.0,
        ]
        model.projection.bias.data = [0.0, 0.0]

        self.assertEqual(generate_text(model, vocab, seed_text="a", length=3), "abab")

    def test_generate_text_errors(self) -> None:
        vocab = CharVocab.from_text("ab")
        model = CharBigramModel(len(vocab), seed=0)

        with self.assertRaises(ValueError):
            generate_text(model, vocab, seed_text="", length=1)

        with self.assertRaises(ValueError):
            generate_text(model, vocab, seed_text="a", length=-1)

        with self.assertRaises(ValueError):
            generate_text(model, vocab, seed_text="z", length=1)

        with self.assertRaises(ValueError):
            CharBigramModel(0)

    def test_state_dict_round_trip(self) -> None:
        model = CharBigramModel(2, seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "char-model.json"
            model.save(path)
            loaded = CharBigramModel(2, seed=1)
            loaded.load(path)

        self.assertEqual(loaded.state_dict(), model.state_dict())

    def test_state_dict_rejects_wrong_vocab_size(self) -> None:
        model = CharBigramModel(2, seed=0)
        with self.assertRaises(ValueError):
            model.load_state_dict({
                "vocab_size": 3,
                "projection": model.projection.state_dict(),
            })

    def test_argmax(self) -> None:
        self.assertEqual(_argmax([1.0, 3.0, 2.0]), 1)
        self.assertEqual(_argmax([3.0, 3.0, 2.0]), 0)

        with self.assertRaises(ValueError):
            _argmax([])

    def test_run_trains_on_tiny_text(self) -> None:
        args = parse_args([
            "--text",
            "abababab",
            "--epochs",
            "20",
            "--batch-size",
            "2",
            "--lr",
            "0.3",
            "--seed-text",
            "a",
            "--generate",
            "4",
        ])
        output = io.StringIO()

        with redirect_stdout(output):
            run(args)

        text = output.getvalue()
        self.assertIn("Character bigram demo", text)
        self.assertIn("vocab size:    2", text)
        self.assertIn("samples:       7", text)
        self.assertIn("generated:", text)


if __name__ == "__main__":
    unittest.main()
