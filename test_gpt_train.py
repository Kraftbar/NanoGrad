import math
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from char_checkpoint import load_checkpoint
from gpt_train import build_model, parse_args, random_batch, split_text, train
from text import CharVocab


TEXT = "First Citizen:\nBefore we proceed any further, hear me speak.\n" * 20


class GptTrainTests(unittest.TestCase):
    def test_split_and_random_batch(self) -> None:
        train_text, validation_text = split_text("abcdefghij", 0.2)
        self.assertEqual((train_text, validation_text), ("abcdefgh", "ij"))

        encoded = list(range(50))
        xs, ys = random_batch(encoded, batch_size=4, context_size=5, rng=random.Random(0))

        self.assertEqual(len(xs), 4)
        for x, y in zip(xs, ys):
            self.assertEqual(len(x), 5)
            self.assertEqual(y, [value + 1 for value in x])

    def test_presets_fill_unset_options(self) -> None:
        args = parse_args(["--text-file", "t.txt", "--preset", "baby", "--steps", "10"])

        self.assertEqual((args.layers, args.heads, args.embedding_dim), (4, 4, 128))
        self.assertEqual(args.steps, 10)

    def test_tiny_run_learns_and_writes_a_loadable_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            text_path = Path(directory) / "text.txt"
            text_path.write_text(TEXT)
            out_dir = Path(directory) / "run"
            args = parse_args([
                "--text-file", str(text_path), "--preset", "mini",
                "--layers", "1", "--embedding-dim", "16", "--context-size", "8",
                "--steps", "30", "--eval-every", "15", "--eval-batches", "2",
                "--generate", "20", "--num-samples", "1", "--out-dir", str(out_dir),
            ])
            with redirect_stdout(StringIO()):
                result = train(args)

            log = result["log"]
            self.assertEqual([row["step"] for row in log], [0, 15, 30])
            self.assertLess(log[-1]["train_loss"], log[0]["train_loss"])
            self.assertTrue((out_dir / "log.csv").exists())
            self.assertTrue((out_dir / "samples.txt").exists())

            vocab = CharVocab.from_text(TEXT)
            model = build_model(len(vocab), args)
            load_checkpoint(out_dir / "model.json", model, vocab, model_name="transformer")


    def test_bpe_run_reports_bits_per_char_and_saves_its_tokenizer(self) -> None:
        from bpe import BPETokenizer

        with tempfile.TemporaryDirectory() as directory:
            text_path = Path(directory) / "text.txt"
            text_path.write_text(TEXT)
            out_dir = Path(directory) / "run"
            args = parse_args([
                "--text-file", str(text_path), "--preset", "mini", "--tokenizer", "bpe",
                "--bpe-vocab-size", "300", "--layers", "1", "--embedding-dim", "16",
                "--context-size", "8", "--steps", "10", "--eval-every", "10", "--eval-batches", "2",
                "--generate", "5", "--num-samples", "1", "--out-dir", str(out_dir),
            ])
            with redirect_stdout(StringIO()):
                result = train(args)

            last = result["log"][-1]
            tokenizer = BPETokenizer.load(out_dir / "tokenizer.json")
            self.assertGreater(len(tokenizer), 256)
            validation = split_text(TEXT, args.validation_fraction)[1]
            chars_per_token = len(validation) / len(tokenizer.encode(validation))
            self.assertAlmostEqual(last["val_bits_per_char"], last["val_loss"] / math.log(2) / chars_per_token)
            model = build_model(len(tokenizer), args)
            model.load(out_dir / "model.json")


    def test_kv_cache_matches_full_recompute_within_one_window(self) -> None:
        import random

        from gpt_train import generate_tokens

        args = parse_args(["--text-file", "t.txt", "--preset", "mini", "--layers", "2",
                           "--embedding-dim", "16", "--context-size", "12"])
        model = build_model(9, args)
        settings = dict(length=10, context_size=12, temperature=1.0, top_k=None)
        full = generate_tokens(model, [1, 2], rng=random.Random(3), kv_cache=False, **settings)
        cached = generate_tokens(model, [1, 2], rng=random.Random(3), kv_cache=True, **settings)
        self.assertEqual(full, cached)
        # Past the window the cache restarts from half a window; still valid tokens.
        longer = generate_tokens(model, [1], rng=random.Random(4), length=40, context_size=12,
                                 temperature=1.0, top_k=None)
        self.assertEqual(len(longer), 41)
        self.assertTrue(all(0 <= token < 9 for token in longer))


    def test_lstm_architecture_trains_and_saves(self) -> None:
        from rnn import CharLSTMModel

        with tempfile.TemporaryDirectory() as directory:
            text_path = Path(directory) / "text.txt"
            text_path.write_text(TEXT)
            out_dir = Path(directory) / "run"
            args = parse_args([
                "--text-file", str(text_path), "--preset", "baby-lstm", "--layers", "1",
                "--embedding-dim", "8", "--lstm-hidden", "16", "--context-size", "8",
                "--steps", "10", "--eval-every", "10", "--eval-batches", "2",
                "--generate", "5", "--num-samples", "1", "--out-dir", str(out_dir),
            ])
            with redirect_stdout(StringIO()):
                result = train(args)
            self.assertLess(result["log"][-1]["train_loss"], result["log"][0]["train_loss"])
            model = build_model(len(CharVocab.from_text(TEXT)), args)
            self.assertIsInstance(model, CharLSTMModel)
            model.load(out_dir / "model.json")


if __name__ == "__main__":
    unittest.main()
