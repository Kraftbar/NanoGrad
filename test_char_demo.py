import io
import json
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from char_demo import (
    _argmax,
    _sample_from_logits,
    apply_checkpoint_config,
    apply_preset,
    build_dataset,
    build_model,
    generate_samples,
    generate_text,
    generation_seed_text,
    generation_input_mode,
    load_checkpoint,
    load_text,
    parse_args,
    run,
    save_checkpoint,
    split_train_validation_text,
    text_source,
)
from language import (
    CharBigramModel,
    CharEmbeddingModel,
    CharTransformerModel,
    TokenPositionEmbedding,
)
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
            "--hidden-dim",
            "8",
            "--heads",
            "2",
            "--layers",
            "2",
            "--activation",
            "gelu",
            "--tie-embeddings",
            "--eval-only",
            "--epochs",
            "3",
            "--batch-size",
            "2",
            "--lr",
            "0.1",
            "--no-shuffle",
            "--optimizer",
            "adam",
            "--weight-decay",
            "0.01",
            "--weight-decay-min-ndim",
            "2",
            "--report-every",
            "2",
            "--validation-chars",
            "3",
            "--max-grad-norm",
            "1.5",
            "--metrics-file",
            "metrics.csv",
            "--output-dir",
            "run",
            "--seed",
            "9",
            "--seed-text",
            "a",
            "--seed-file",
            "prompt.txt",
            "--generate",
            "5",
            "--num-samples",
            "2",
            "--samples-file",
            "samples.csv",
            "--generate-only",
            "--sample-mode",
            "sample",
            "--temperature",
            "0.8",
            "--top-k",
            "3",
            "--sample-seed",
            "4",
            "--save-model",
            "char-model.json",
            "--load-model",
            "char-model-in.json",
        ])

        self.assertEqual(args.model, "embedding")
        self.assertEqual(args.text, "abba")
        self.assertEqual(args.text_file, Path("tiny.txt"))
        self.assertEqual(args.max_chars, 3)
        self.assertEqual(args.context_size, 2)
        self.assertEqual(args.embedding_dim, 4)
        self.assertEqual(args.hidden_dim, 8)
        self.assertEqual(args.heads, 2)
        self.assertEqual(args.layers, 2)
        self.assertEqual(args.activation, "gelu")
        self.assertTrue(args.tie_embeddings)
        self.assertTrue(args.eval_only)
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.batch_size, 2)
        self.assertEqual(args.lr, 0.1)
        self.assertTrue(args.no_shuffle)
        self.assertEqual(args.optimizer, "adam")
        self.assertEqual(args.weight_decay, 0.01)
        self.assertEqual(args.weight_decay_min_ndim, 2)
        self.assertEqual(args.report_every, 2)
        self.assertEqual(args.validation_chars, 3)
        self.assertEqual(args.max_grad_norm, 1.5)
        self.assertEqual(args.metrics_file, Path("metrics.csv"))
        self.assertEqual(args.output_dir, Path("run"))
        self.assertEqual(args.seed, 9)
        self.assertEqual(args.seed_text, "a")
        self.assertEqual(args.seed_file, Path("prompt.txt"))
        self.assertEqual(args.generate, 5)
        self.assertEqual(args.num_samples, 2)
        self.assertEqual(args.samples_file, Path("samples.csv"))
        self.assertTrue(args.generate_only)
        self.assertEqual(args.sample_mode, "sample")
        self.assertEqual(args.temperature, 0.8)
        self.assertEqual(args.top_k, 3)
        self.assertEqual(args.sample_seed, 4)
        self.assertEqual(args.save_model, Path("char-model.json"))
        self.assertEqual(args.load_model, Path("char-model-in.json"))

    def test_tiny_transformer_preset_sets_smoke_defaults(self) -> None:
        args = parse_args(["--preset", "tiny-transformer"])

        self.assertEqual(args.preset, "tiny-transformer")
        self.assertEqual(args.model, "transformer")
        self.assertEqual(args.max_chars, 128)
        self.assertEqual(args.context_size, 4)
        self.assertEqual(args.embedding_dim, 8)
        self.assertEqual(args.hidden_dim, 16)
        self.assertEqual(args.heads, 1)
        self.assertEqual(args.layers, 1)
        self.assertEqual(args.activation, "relu")
        self.assertFalse(args.tie_embeddings)
        self.assertEqual(args.epochs, 1)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.lr, 0.05)
        self.assertEqual(args.seed_text, "hell")
        self.assertEqual(args.generate, 24)

    def test_tiny_gpt_preset_sets_gpt_like_smoke_defaults(self) -> None:
        args = parse_args(["--preset", "tiny-gpt"])

        self.assertEqual(args.preset, "tiny-gpt")
        self.assertEqual(args.model, "transformer")
        self.assertEqual(args.max_chars, 128)
        self.assertEqual(args.context_size, 4)
        self.assertEqual(args.embedding_dim, 8)
        self.assertEqual(args.hidden_dim, 16)
        self.assertEqual(args.heads, 1)
        self.assertEqual(args.layers, 1)
        self.assertEqual(args.activation, "gelu")
        self.assertTrue(args.tie_embeddings)
        self.assertEqual(args.epochs, 1)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.lr, 0.01)
        self.assertEqual(args.optimizer, "adam")
        self.assertEqual(args.weight_decay, 0.01)
        self.assertEqual(args.weight_decay_min_ndim, 2)
        self.assertEqual(args.max_grad_norm, 1.0)
        self.assertEqual(args.seed_text, "hell")
        self.assertEqual(args.generate, 24)

    def test_tiny_gpt_preset_keeps_explicit_false_boolean(self) -> None:
        args = parse_args([
            "--preset",
            "tiny-gpt",
            "--no-tie-embeddings",
            "--optimizer",
            "sgd",
        ])

        self.assertFalse(args.tie_embeddings)
        self.assertEqual(args.optimizer, "sgd")
        self.assertEqual(args.activation, "gelu")

    def test_lite_gpt_preset_sets_middle_experiment_defaults(self) -> None:
        args = parse_args(["--preset", "lite-gpt"])

        self.assertEqual(args.preset, "lite-gpt")
        self.assertEqual(args.model, "transformer")
        self.assertEqual(args.max_chars, 256)
        self.assertEqual(args.context_size, 8)
        self.assertEqual(args.embedding_dim, 12)
        self.assertEqual(args.hidden_dim, 32)
        self.assertEqual(args.heads, 1)
        self.assertEqual(args.layers, 1)
        self.assertEqual(args.activation, "gelu")
        self.assertTrue(args.tie_embeddings)
        self.assertEqual(args.epochs, 1)
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.lr, 0.0075)
        self.assertEqual(args.optimizer, "adam")
        self.assertEqual(args.weight_decay, 0.01)
        self.assertEqual(args.weight_decay_min_ndim, 2)
        self.assertEqual(args.max_grad_norm, 1.0)
        self.assertEqual(args.seed_text, "hello na")
        self.assertEqual(args.generate, 64)

    def test_small_gpt_preset_sets_larger_experiment_defaults(self) -> None:
        args = parse_args(["--preset", "small-gpt"])

        self.assertEqual(args.preset, "small-gpt")
        self.assertEqual(args.model, "transformer")
        self.assertEqual(args.max_chars, 512)
        self.assertEqual(args.context_size, 8)
        self.assertEqual(args.embedding_dim, 16)
        self.assertEqual(args.hidden_dim, 64)
        self.assertEqual(args.heads, 2)
        self.assertEqual(args.layers, 2)
        self.assertEqual(args.activation, "gelu")
        self.assertTrue(args.tie_embeddings)
        self.assertEqual(args.epochs, 1)
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.lr, 0.005)
        self.assertEqual(args.optimizer, "adam")
        self.assertEqual(args.weight_decay, 0.01)
        self.assertEqual(args.weight_decay_min_ndim, 2)
        self.assertEqual(args.max_grad_norm, 1.0)
        self.assertEqual(args.seed_text, "hello na")
        self.assertEqual(args.generate, 80)

    def test_tiny_transformer_preset_keeps_explicit_values(self) -> None:
        args = parse_args([
            "--preset",
            "tiny-transformer",
            "--embedding-dim",
            "6",
            "--max-chars",
            "32",
            "--seed-text",
            "abcd",
            "--generate",
            "2",
        ])

        self.assertEqual(args.model, "transformer")
        self.assertEqual(args.context_size, 4)
        self.assertEqual(args.embedding_dim, 6)
        self.assertEqual(args.max_chars, 32)
        self.assertEqual(args.seed_text, "abcd")
        self.assertEqual(args.generate, 2)

        args.preset = None
        self.assertIs(apply_preset(args), args)

    def test_tiny_transformer_preset_keeps_explicit_default_values(self) -> None:
        args = parse_args([
            "--preset",
            "tiny-transformer",
            "--model",
            "bigram",
            "--context-size",
            "1",
            "--seed-text",
            "h",
        ])

        self.assertEqual(args.model, "bigram")
        self.assertEqual(args.context_size, 1)
        self.assertEqual(args.seed_text, "h")
        self.assertEqual(args.embedding_dim, 8)

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
        transformer_args = parse_args([
            "--model",
            "transformer",
            "--context-size",
            "2",
            "--embedding-dim",
            "4",
            "--hidden-dim",
            "8",
            "--heads",
            "2",
            "--activation",
            "gelu",
            "--tie-embeddings",
        ])

        bigram_dataset, bigram_vocab = build_dataset("abca", bigram_args)
        embedding_dataset, embedding_vocab = build_dataset("abca", embedding_args)
        transformer_dataset, transformer_vocab = build_dataset(
            "abca",
            transformer_args,
        )

        self.assertEqual(len(bigram_vocab), 3)
        self.assertEqual(len(embedding_vocab), 3)
        self.assertEqual(len(transformer_vocab), 3)
        self.assertEqual(bigram_dataset.feature_shape, (6,))
        self.assertEqual(embedding_dataset.feature_shape, (2,))
        self.assertEqual(transformer_dataset.feature_shape, (2,))
        self.assertEqual(transformer_dataset.target_shape, (2,))
        self.assertIsInstance(build_model(bigram_args, vocab_size=3), CharBigramModel)
        self.assertIsInstance(
            build_model(embedding_args, vocab_size=3),
            CharEmbeddingModel,
        )
        transformer_model = build_model(transformer_args, vocab_size=3)
        self.assertIsInstance(transformer_model, CharTransformerModel)
        self.assertEqual(transformer_model.feed_forward_activation, "gelu")
        self.assertTrue(transformer_model.tie_embeddings)
        self.assertIs(
            transformer_model.projection.weight,
            transformer_model.embedding.token_embedding.weight,
        )

    def test_generation_input_mode(self) -> None:
        self.assertEqual(generation_input_mode("bigram"), "bigram")
        self.assertEqual(generation_input_mode("embedding"), "embedding")
        self.assertEqual(generation_input_mode("transformer"), "embedding")

        with self.assertRaises(ValueError):
            generation_input_mode("unknown")

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

    def test_generate_samples_reuses_rng_across_multiple_samples(self) -> None:
        vocab = CharVocab.from_text("ab")
        model = CharBigramModel(len(vocab), seed=0)
        model.projection.weight.data = [0.0, 0.0, 0.0, 0.0]
        model.projection.bias.data = [0.0, 0.0]
        args = parse_args([
            "--sample-mode",
            "sample",
            "--sample-seed",
            "0",
            "--num-samples",
            "2",
            "--seed-text",
            "a",
            "--generate",
            "3",
        ])

        self.assertEqual(
            generate_samples(model, vocab, args),
            ["abba", "aaba"],
        )

        args.num_samples = 0
        with self.assertRaises(ValueError):
            generate_samples(model, vocab, args)

    def test_generation_seed_text_can_read_seed_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prompt.txt"
            path.write_text("from file", encoding="utf-8")
            args = parse_args([
                "--seed-text",
                "from arg",
                "--seed-file",
                str(path),
            ])

            self.assertEqual(generation_seed_text(args), "from file")

            path.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                generation_seed_text(args)

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
            generate_text(
                model,
                vocab,
                seed_text="a",
                length=1,
                sample_mode="sample",
                top_k=0,
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
        self.assertEqual(
            _sample_from_logits([3.0, 2.0, 1.0], top_k=1, rng=random.Random(0)),
            0,
        )

        with self.assertRaises(ValueError):
            _sample_from_logits([])

        with self.assertRaises(ValueError):
            _sample_from_logits([1.0], temperature=0.0)

        with self.assertRaises(ValueError):
            _sample_from_logits([1.0], top_k=0)

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

    def test_split_train_validation_text(self) -> None:
        args = parse_args([
            "--text",
            "abcdef",
            "--validation-chars",
            "2",
        ])

        self.assertEqual(
            split_train_validation_text("abcdef", args),
            ("abcd", "ef"),
        )

        args = parse_args(["--validation-chars", "0"])
        self.assertEqual(split_train_validation_text("abcdef", args), ("abcdef", None))

        args = parse_args(["--validation-chars", "-1"])
        with self.assertRaises(ValueError):
            split_train_validation_text("abcdef", args)

        args = parse_args(["--validation-chars", "6"])
        with self.assertRaises(ValueError):
            split_train_validation_text("abcdef", args)

        args = parse_args(["--context-size", "2", "--validation-chars", "1"])
        with self.assertRaises(ValueError):
            split_train_validation_text("abcd", args)

    def test_run_trains_on_tiny_text(self) -> None:
        args = parse_args([
            "--text",
            "abababab",
            "--epochs",
            "2",
            "--batch-size",
            "2",
            "--lr",
            "0.3",
            "--no-shuffle",
            "--optimizer",
            "adam",
            "--weight-decay",
            "0.01",
            "--report-every",
            "1",
            "--max-grad-norm",
            "1.0",
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
        self.assertIn("sample dist-2:", text)
        self.assertIn("text source:   built-in", text)
        self.assertIn("vocab size:    2", text)
        self.assertIn("samples:       7", text)
        self.assertIn("max grad norm: 1.0", text)
        self.assertIn("optimizer:     adam", text)
        self.assertIn("shuffle:       False", text)
        self.assertIn("weight decay:  0.01", text)
        self.assertIn("epoch 1/2", text)
        self.assertIn("epoch 2/2", text)
        self.assertIn("generated:", text)

    def test_run_trains_with_validation_split(self) -> None:
        args = parse_args([
            "--text",
            "ababababab",
            "--epochs",
            "2",
            "--batch-size",
            "2",
            "--lr",
            "0.3",
            "--report-every",
            "1",
            "--validation-chars",
            "4",
            "--seed-text",
            "a",
            "--generate",
            "2",
        ])
        output = io.StringIO()

        with redirect_stdout(output):
            run(args)

        text = output.getvalue()
        self.assertIn("val samples:", text)
        self.assertIn("val_loss=", text)
        self.assertIn("val_ppl=", text)
        self.assertIn("val loss:", text)
        self.assertIn("val ppl:", text)
        self.assertIn("val accuracy:", text)

    def test_run_writes_epoch_metrics_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "char-metrics.csv"
            args = parse_args([
                "--text",
                "ababababab",
                "--epochs",
                "2",
                "--batch-size",
                "2",
                "--lr",
                "0.3",
                "--validation-chars",
                "4",
                "--seed-text",
                "a",
                "--generate",
                "2",
                "--metrics-file",
                str(metrics_path),
            ])
            output = io.StringIO()

            with redirect_stdout(output):
                run(args)

            metrics = metrics_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                metrics[0],
                "epoch,loss,perplexity,accuracy,val_loss,val_perplexity,val_accuracy,elapsed_seconds,examples_seen",
            )
            self.assertEqual(len(metrics), 3)
            self.assertTrue(metrics[1].startswith("1,"))
            self.assertTrue(metrics[2].startswith("2,"))
            self.assertIn("metrics file:", output.getvalue())

    def test_run_output_dir_writes_standard_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            args = parse_args([
                "--text",
                "ababababab",
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--lr",
                "0.3",
                "--validation-chars",
                "4",
                "--seed-text",
                "a",
                "--generate",
                "2",
                "--output-dir",
                str(output_dir),
            ])
            output = io.StringIO()

            with redirect_stdout(output):
                run(args)

            model = json.loads((output_dir / "model.json").read_text(encoding="utf-8"))
            metrics = (output_dir / "metrics.csv").read_text(encoding="utf-8")
            samples = (output_dir / "samples.csv").read_text(encoding="utf-8")

        text = output.getvalue()
        self.assertIn("output dir:", text)
        self.assertIn("saved model:", text)
        self.assertIn("metrics file:", text)
        self.assertIn("samples file:", text)
        self.assertEqual(model["format"], "nanollm.char_demo.v1")
        self.assertIn("epoch,loss,perplexity", metrics)
        self.assertIn("sample_index,seed_text", samples)

    def test_run_saves_and_loads_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "char-model.json"
            reload_path = Path(tmpdir) / "char-model-reloaded.json"
            args = parse_args([
                "--text",
                "abababab",
                "--epochs",
                "2",
                "--batch-size",
                "2",
                "--lr",
                "0.3",
                "--seed-text",
                "a",
                "--generate",
                "2",
                "--save-model",
                str(save_path),
            ])
            with redirect_stdout(io.StringIO()):
                run(args)

            reload_args = parse_args([
                "--text",
                "abababab",
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--lr",
                "0.3",
                "--seed-text",
                "a",
                "--generate",
                "2",
                "--load-model",
                str(save_path),
                "--save-model",
                str(reload_path),
            ])
            output = io.StringIO()
            with redirect_stdout(output):
                run(reload_args)

            self.assertTrue(save_path.exists())
            self.assertTrue(reload_path.exists())
            payload = json.loads(save_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["format"], "nanollm.char_demo.v1")
            self.assertEqual(payload["model"], "bigram")
            self.assertEqual(payload["vocab"], ["a", "b"])
            self.assertIn("saved model:", output.getvalue())

    def test_checkpoint_rejects_wrong_vocabulary(self) -> None:
        vocab = CharVocab.from_text("ab")
        model = CharBigramModel(len(vocab), seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "char-model.json"
            save_checkpoint(path, model, vocab, model_name="bigram")

            target_vocab = CharVocab.from_text("cd")
            target = CharBigramModel(len(target_vocab), seed=1)
            with self.assertRaises(ValueError):
                load_checkpoint(
                    path,
                    target,
                    target_vocab,
                    model_name="bigram",
                )

    def test_checkpoint_loads_raw_model_state(self) -> None:
        vocab = CharVocab.from_text("ab")
        model = CharBigramModel(len(vocab), seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "char-model-raw.json"
            model.save(path)
            target = CharBigramModel(len(vocab), seed=1)
            load_checkpoint(path, target, vocab, model_name="bigram")

        self.assertEqual(target.state_dict(), model.state_dict())

    def test_run_eval_only_reports_loaded_model_without_training(self) -> None:
        vocab = CharVocab.from_text("ab")
        model = CharBigramModel(len(vocab), seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "char-model.json"
            save_checkpoint(path, model, vocab, model_name="bigram")
            args = parse_args([
                "--text",
                "abababab",
                "--load-model",
                str(path),
                "--eval-only",
                "--epochs",
                "20",
                "--seed-text",
                "a",
                "--generate",
                "2",
            ])
            output = io.StringIO()

            with redirect_stdout(output):
                run(args)

        text = output.getvalue()
        self.assertIn("eval only:     True", text)
        self.assertIn("train loss:", text)
        self.assertIn("train ppl:", text)
        self.assertNotIn("final batch:", text)

    def test_load_model_applies_checkpoint_architecture_defaults(self) -> None:
        vocab = CharVocab.from_text("ab")
        model = CharTransformerModel(
            len(vocab),
            context_size=2,
            embedding_dim=4,
            hidden_dim=8,
            num_heads=2,
            feed_forward_activation="gelu",
            tie_embeddings=True,
            seed=0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "char-transformer.json"
            save_checkpoint(path, model, vocab, model_name="transformer")
            args = parse_args([
                "--text",
                "abababab",
                "--load-model",
                str(path),
                "--eval-only",
                "--seed-text",
                "ab",
                "--generate",
                "1",
            ])
            output = io.StringIO()

            with redirect_stdout(output):
                run(args)

        text = output.getvalue()
        self.assertIn("model:         transformer", text)
        self.assertIn("context size:  2", text)
        self.assertIn("embedding dim: 4", text)
        self.assertIn("hidden dim:    8", text)
        self.assertIn("heads:         2", text)
        self.assertIn("activation:    gelu", text)
        self.assertIn("tie embeddings: True", text)

    def test_run_generate_only_uses_checkpoint_vocab_and_architecture(self) -> None:
        vocab = CharVocab.from_text("ab")
        model = CharTransformerModel(
            len(vocab),
            context_size=2,
            embedding_dim=4,
            hidden_dim=8,
            num_heads=2,
            feed_forward_activation="gelu",
            tie_embeddings=True,
            seed=0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "char-transformer.json"
            samples_path = Path(tmpdir) / "samples.csv"
            save_checkpoint(path, model, vocab, model_name="transformer")
            args = parse_args([
                "--load-model",
                str(path),
                "--generate-only",
                "--seed-text",
                "ab",
                "--generate",
                "1",
                "--num-samples",
                "2",
                "--samples-file",
                str(samples_path),
            ])
            output = io.StringIO()

            with redirect_stdout(output):
                run(args)

            samples = samples_path.read_text(encoding="utf-8").splitlines()

        text = output.getvalue()
        self.assertIn("model:         transformer", text)
        self.assertIn("generate only: True", text)
        self.assertIn("num samples:   2", text)
        self.assertIn("checkpoint:", text)
        self.assertIn("vocab size:    2", text)
        self.assertIn("context size:  2", text)
        self.assertIn("hidden dim:    8", text)
        self.assertIn("generated 1:", text)
        self.assertIn("generated 2:", text)
        self.assertIn("sample dist-2:", text)
        self.assertIn("samples file:", text)
        self.assertNotIn("text length:", text)
        self.assertNotIn("initial loss:", text)
        self.assertEqual(
            samples[0],
            "sample_index,seed_text,sample_mode,temperature,top_k,distinct_2,sample",
        )
        self.assertEqual(len(samples), 3)
        self.assertTrue(samples[1].startswith("1,ab,greedy,1.0,"))

    def test_generate_only_requires_checkpoint_and_rejects_metrics(self) -> None:
        args = parse_args(["--generate-only"])
        with self.assertRaises(ValueError):
            run(args)

        vocab = CharVocab.from_text("ab")
        model = CharBigramModel(len(vocab), seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "char-model.json"
            metrics_path = Path(tmpdir) / "metrics.csv"
            save_checkpoint(path, model, vocab, model_name="bigram")
            args = parse_args([
                "--load-model",
                str(path),
                "--generate-only",
                "--metrics-file",
                str(metrics_path),
            ])
            with self.assertRaises(ValueError):
                run(args)

    def test_checkpoint_config_keeps_explicit_architecture_overrides(self) -> None:
        vocab = CharVocab.from_text("ab")
        model = CharBigramModel(len(vocab), context_size=2, seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "char-model.json"
            save_checkpoint(path, model, vocab, model_name="bigram")
            args = parse_args([
                "--load-model",
                str(path),
                "--context-size",
                "1",
            ])

            apply_checkpoint_config(args, path)

        self.assertEqual(args.model, "bigram")
        self.assertEqual(args.context_size, 1)

    def test_checkpoint_config_defaults_older_transformer_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy-transformer.json"
            path.write_text(
                json.dumps({
                    "format": "nanollm.char_demo.v1",
                    "model": "transformer",
                    "vocab": ["a", "b"],
                    "state": {
                        "context_size": 2,
                        "embedding_dim": 4,
                        "hidden_dim": 8,
                        "num_layers": 1,
                    },
                }),
                encoding="utf-8",
            )
            args = parse_args(["--load-model", str(path)])

            apply_checkpoint_config(args, path)

        self.assertEqual(args.model, "transformer")
        self.assertEqual(args.context_size, 2)
        self.assertEqual(args.embedding_dim, 4)
        self.assertEqual(args.hidden_dim, 8)
        self.assertEqual(args.heads, 1)
        self.assertEqual(args.activation, "relu")
        self.assertFalse(args.tie_embeddings)

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
            "--top-k",
            "1",
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
        self.assertIn("top k:         1", text)
        self.assertIn("generated:", text)

    def test_run_trains_transformer_model_on_tiny_text(self) -> None:
        args = parse_args([
            "--model",
            "transformer",
            "--text",
            "abababab",
            "--context-size",
            "2",
            "--embedding-dim",
            "4",
            "--hidden-dim",
            "8",
            "--heads",
            "2",
            "--activation",
            "gelu",
            "--tie-embeddings",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--lr",
            "0.1",
            "--seed-text",
            "ab",
            "--generate",
            "2",
        ])
        output = io.StringIO()

        with redirect_stdout(output):
            run(args)

        text = output.getvalue()
        self.assertIn("model:         transformer", text)
        self.assertIn("context size:  2", text)
        self.assertIn("embedding dim: 4", text)
        self.assertIn("hidden dim:    8", text)
        self.assertIn("heads:         2", text)
        self.assertIn("layers:        1", text)
        self.assertIn("activation:    gelu", text)
        self.assertIn("tie embeddings: True", text)
        self.assertIn("objective:     sequence", text)
        self.assertIn("generated:", text)


if __name__ == "__main__":
    unittest.main()
