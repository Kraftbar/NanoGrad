import io
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from char_demo import (
    _argmax,
    _sample_from_logits,
    build_dataset,
    build_model,
    generate_text,
    load_text,
    parse_args,
    run,
    text_source,
)
from language import CharBigramModel, CharEmbeddingModel, TokenPositionEmbedding
from tensor import Tensor
from text import CharVocab


class CharDemoTests(unittest.TestCase):
    def test_parse_args(self) -> None:
        args = parse_args([
            "--model",
            "embedding",
            "--text",
            "abba",
            "--text-file",
            "tiny.txt",
            "--max-chars",
            "3",
            "--context-size",
            "2",
            "--embedding-dim",
            "4",
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
            "--sample-mode",
            "sample",
            "--temperature",
            "0.8",
            "--sample-seed",
            "4",
        ])

        self.assertEqual(args.model, "embedding")
        self.assertEqual(args.text, "abba")
        self.assertEqual(args.text_file, Path("tiny.txt"))
        self.assertEqual(args.max_chars, 3)
        self.assertEqual(args.context_size, 2)
        self.assertEqual(args.embedding_dim, 4)
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.batch_size, 2)
        self.assertEqual(args.lr, 0.1)
        self.assertEqual(args.seed, 9)
        self.assertEqual(args.seed_text, "a")
        self.assertEqual(args.generate, 5)
        self.assertEqual(args.sample_mode, "sample")
        self.assertEqual(args.temperature, 0.8)
        self.assertEqual(args.sample_seed, 4)

    def test_char_bigram_model_forward_shape(self) -> None:
        model = CharBigramModel(3, context_size=2, seed=0)

        logits = model(Tensor.from_list([
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ]))

        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(model.num_parameters(), 21)

    def test_char_embedding_model_forward_shape(self) -> None:
        model = CharEmbeddingModel(
            3,
            context_size=2,
            embedding_dim=4,
            seed=0,
        )

        logits = model(Tensor.from_list([
            [0, 1],
            [1, 2],
        ]))

        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(model.num_parameters(), 47)

    def test_token_position_embedding_adds_positions(self) -> None:
        layer = TokenPositionEmbedding(
            vocab_size=3,
            context_size=2,
            embedding_dim=2,
            seed=0,
        )
        layer.token_embedding.weight.data = [
            1.0,
            10.0,
            2.0,
            20.0,
            3.0,
            30.0,
        ]
        layer.position_embedding.weight.data = [
            0.5,
            1.0,
            -0.5,
            -1.0,
        ]

        outputs = layer(Tensor.from_list([
            [0, 1],
            [2, 0],
        ]))

        self.assertEqual(outputs.shape, (2, 2, 2))
        self.assertEqual(
            outputs.tolist(),
            [
                [
                    [1.5, 11.0],
                    [1.5, 19.0],
                ],
                [
                    [3.5, 31.0],
                    [0.5, 9.0],
                ],
            ],
        )
        self.assertEqual(layer.num_parameters(), 10)

    def test_build_dataset_and_model_use_requested_type(self) -> None:
        bigram_args = parse_args(["--model", "bigram", "--context-size", "2"])
        embedding_args = parse_args(["--model", "embedding", "--context-size", "2"])

        bigram_dataset, bigram_vocab = build_dataset("abca", bigram_args)
        embedding_dataset, embedding_vocab = build_dataset("abca", embedding_args)

        self.assertEqual(len(bigram_vocab), 3)
        self.assertEqual(len(embedding_vocab), 3)
        self.assertEqual(bigram_dataset.feature_shape, (6,))
        self.assertEqual(embedding_dataset.feature_shape, (2,))
        self.assertIsInstance(build_model(bigram_args, vocab_size=3), CharBigramModel)
        self.assertIsInstance(
            build_model(embedding_args, vocab_size=3),
            CharEmbeddingModel,
        )

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

    def test_generate_text_supports_embedding_model(self) -> None:
        vocab = CharVocab.from_text("ab")
        model = CharEmbeddingModel(
            len(vocab),
            context_size=1,
            embedding_dim=2,
            seed=0,
        )
        model.embedding.token_embedding.weight.data = [
            1.0,
            0.0,
            0.0,
            1.0,
        ]
        model.embedding.position_embedding.weight.data = [0.0, 0.0]
        model.projection.weight.data = [
            0.0,
            1.0,
            2.0,
            0.0,
        ]
        model.projection.bias.data = [0.0, 0.0]

        self.assertEqual(
            generate_text(
                model,
                vocab,
                seed_text="a",
                length=3,
                input_mode="embedding",
            ),
            "abab",
        )

    def test_generate_text_can_sample_with_seed(self) -> None:
        vocab = CharVocab.from_text("ab")
        model = CharBigramModel(len(vocab), seed=0)
        model.projection.weight.data = [0.0, 0.0, 0.0, 0.0]
        model.projection.bias.data = [0.0, 0.0]

        self.assertEqual(
            generate_text(
                model,
                vocab,
                seed_text="a",
                length=3,
                sample_mode="sample",
                rng=random.Random(0),
            ),
            "abba",
        )

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
            generate_text(model, vocab, seed_text="a", length=1, context_size=2)

        with self.assertRaises(ValueError):
            generate_text(
                model,
                vocab,
                seed_text="a",
                length=1,
                input_mode="unknown",
            )

        with self.assertRaises(ValueError):
            generate_text(
                model,
                vocab,
                seed_text="a",
                length=1,
                sample_mode="unknown",
            )

        with self.assertRaises(ValueError):
            generate_text(
                model,
                vocab,
                seed_text="a",
                length=1,
                sample_mode="sample",
                temperature=0.0,
            )

        with self.assertRaises(ValueError):
            CharBigramModel(0)

        with self.assertRaises(ValueError):
            CharBigramModel(2, context_size=0)

        with self.assertRaises(ValueError):
            CharEmbeddingModel(0)

        with self.assertRaises(ValueError):
            CharEmbeddingModel(2, context_size=0)

        with self.assertRaises(ValueError):
            CharEmbeddingModel(2, embedding_dim=0)

        with self.assertRaises(ValueError):
            TokenPositionEmbedding(0, context_size=1, embedding_dim=1)

        with self.assertRaises(ValueError):
            TokenPositionEmbedding(2, context_size=0, embedding_dim=1)

        with self.assertRaises(ValueError):
            TokenPositionEmbedding(2, context_size=1, embedding_dim=0)

        with self.assertRaises(ValueError):
            TokenPositionEmbedding(2, context_size=2, embedding_dim=1)(
                Tensor.from_list([[0, 1, 0]]),
            )

    def test_state_dict_round_trip(self) -> None:
        model = CharBigramModel(2, seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "char-model.json"
            model.save(path)
            loaded = CharBigramModel(2, seed=1)
            loaded.load(path)

        self.assertEqual(loaded.state_dict(), model.state_dict())

    def test_embedding_state_dict_round_trip(self) -> None:
        model = CharEmbeddingModel(2, context_size=2, embedding_dim=3, seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "char-embedding-model.json"
            model.save(path)
            loaded = CharEmbeddingModel(2, context_size=2, embedding_dim=3, seed=1)
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

    def test_sample_from_logits(self) -> None:
        self.assertEqual(
            _sample_from_logits([0.0, 0.0], rng=random.Random(0)),
            1,
        )
        self.assertEqual(
            _sample_from_logits([0.0, 4.0], rng=random.Random(0)),
            1,
        )

        with self.assertRaises(ValueError):
            _sample_from_logits([])

        with self.assertRaises(ValueError):
            _sample_from_logits([1.0], temperature=0.0)

    def test_load_text_prefers_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "input.txt"
            path.write_text("from file", encoding="utf-8")
            args = parse_args([
                "--text",
                "from arg",
                "--text-file",
                str(path),
            ])

            self.assertEqual(load_text(args), "from file")
            self.assertEqual(text_source(args), str(path))

    def test_load_text_can_cap_character_count(self) -> None:
        args = parse_args([
            "--text",
            "abcdef",
            "--max-chars",
            "3",
        ])

        self.assertEqual(load_text(args), "abc")

        args = parse_args(["--max-chars", "0"])
        with self.assertRaises(ValueError):
            load_text(args)

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
        self.assertIn("Character language demo", text)
        self.assertIn("model:         bigram", text)
        self.assertIn("generation:    greedy", text)
        self.assertIn("text source:   built-in", text)
        self.assertIn("vocab size:    2", text)
        self.assertIn("samples:       7", text)
        self.assertIn("generated:", text)

    def test_run_trains_on_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "input.txt"
            path.write_text("abababab", encoding="utf-8")
            args = parse_args([
                "--text-file",
                str(path),
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
        self.assertIn(f"text source:   {path}", text)
        self.assertIn("vocab size:    2", text)

    def test_run_trains_embedding_model_on_tiny_text(self) -> None:
        args = parse_args([
            "--model",
            "embedding",
            "--text",
            "abababab",
            "--context-size",
            "2",
            "--embedding-dim",
            "4",
            "--epochs",
            "20",
            "--batch-size",
            "2",
            "--lr",
            "0.3",
            "--seed-text",
            "ab",
            "--generate",
            "4",
            "--sample-mode",
            "sample",
            "--temperature",
            "0.9",
            "--sample-seed",
            "7",
        ])
        output = io.StringIO()

        with redirect_stdout(output):
            run(args)

        text = output.getvalue()
        self.assertIn("model:         embedding", text)
        self.assertIn("context size:  2", text)
        self.assertIn("embedding dim: 4", text)
        self.assertIn("generation:    sample", text)
        self.assertIn("temperature:   0.9", text)
        self.assertIn("generated:", text)


if __name__ == "__main__":
    unittest.main()
