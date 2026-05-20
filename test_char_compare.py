import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from char_compare import (
    ComparisonResult,
    format_results_table,
    parse_args,
    perplexity,
    run,
    usable_seed_text,
)
from text import CharVocab


class CharCompareTests(unittest.TestCase):
    def test_parse_args(self) -> None:
        args = parse_args([
            "--models",
            "bigram",
            "tiny-gpt",
            "--text",
            "abcdabcd",
            "--text-file",
            "input.txt",
            "--max-chars",
            "16",
            "--validation-chars",
            "5",
            "--epochs",
            "2",
            "--batch-size",
            "3",
            "--lr",
            "0.1",
            "--optimizer",
            "adam",
            "--weight-decay",
            "0.01",
            "--weight-decay-min-ndim",
            "2",
            "--max-grad-norm",
            "1.0",
            "--no-shuffle",
            "--seed",
            "7",
            "--seed-text",
            "abcd",
            "--seed-file",
            "prompt.txt",
            "--generate",
            "4",
            "--num-samples",
            "2",
            "--sample-mode",
            "sample",
            "--temperature",
            "0.8",
            "--top-k",
            "2",
            "--sample-seed",
            "11",
            "--summary-file",
            "summary.csv",
            "--metrics-file",
            "metrics.csv",
            "--samples-file",
            "samples.csv",
        ])

        self.assertEqual(args.models, ["bigram", "tiny-gpt"])
        self.assertEqual(args.text, "abcdabcd")
        self.assertEqual(args.text_file, Path("input.txt"))
        self.assertEqual(args.max_chars, 16)
        self.assertEqual(args.validation_chars, 5)
        self.assertEqual(args.epochs, 2)
        self.assertEqual(args.batch_size, 3)
        self.assertEqual(args.lr, 0.1)
        self.assertEqual(args.optimizer, "adam")
        self.assertEqual(args.weight_decay, 0.01)
        self.assertEqual(args.weight_decay_min_ndim, 2)
        self.assertEqual(args.max_grad_norm, 1.0)
        self.assertTrue(args.no_shuffle)
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.seed_text, "abcd")
        self.assertEqual(args.seed_file, Path("prompt.txt"))
        self.assertEqual(args.generate, 4)
        self.assertEqual(args.num_samples, 2)
        self.assertEqual(args.sample_mode, "sample")
        self.assertEqual(args.temperature, 0.8)
        self.assertEqual(args.top_k, 2)
        self.assertEqual(args.sample_seed, 11)
        self.assertEqual(args.summary_file, Path("summary.csv"))
        self.assertEqual(args.metrics_file, Path("metrics.csv"))
        self.assertEqual(args.samples_file, Path("samples.csv"))

    def test_default_models_skip_slower_small_gpt(self) -> None:
        args = parse_args([])

        self.assertEqual(args.models, ["bigram", "embedding", "tiny-gpt"])
        self.assertEqual(args.validation_chars, 9)

    def test_small_gpt_can_be_selected_explicitly(self) -> None:
        args = parse_args(["--models", "small-gpt"])

        self.assertEqual(args.models, ["small-gpt"])
        self.assertEqual(args.validation_chars, 9)

    def test_usable_seed_text_falls_back_to_training_prefix(self) -> None:
        vocab = CharVocab.from_text("abab")

        self.assertEqual(
            usable_seed_text("zzzz", train_text="abab", vocab=vocab, context_size=2),
            "ab",
        )
        self.assertEqual(
            usable_seed_text("baba", train_text="abab", vocab=vocab, context_size=2),
            "baba",
        )

    def test_format_results_table(self) -> None:
        table = format_results_table([
            ComparisonResult(
                name="bigram",
                model="bigram",
                context_size=4,
                parameters=10,
                train_loss=1.0,
                train_perplexity=perplexity(1.0),
                accuracy=0.5,
                validation_loss=1.2,
                validation_perplexity=perplexity(1.2),
                validation_accuracy=0.25,
                elapsed_seconds=0.1,
                examples_per_second=20.0,
                sample_distinct_2=0.75,
                generated_samples=["abcd"],
            ),
        ])

        self.assertIn("name", table)
        self.assertIn("ctx", table)
        self.assertIn("train ppl", table)
        self.assertIn("dist-2", table)
        self.assertIn("bigram", table)

    def test_run_compares_models_and_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.csv"
            metrics_path = Path(tmpdir) / "metrics.csv"
            samples_path = Path(tmpdir) / "samples.csv"
            args = parse_args([
                "--models",
                "bigram",
                "embedding",
                "tiny-gpt",
                "--text",
                "abcdabcdabcd",
                "--validation-chars",
                "5",
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--seed-text",
                "abcd",
                "--generate",
                "2",
                "--num-samples",
                "2",
                "--summary-file",
                str(summary_path),
                "--metrics-file",
                str(metrics_path),
                "--samples-file",
                str(samples_path),
            ])
            output = io.StringIO()

            with redirect_stdout(output):
                run(args)

            summary = summary_path.read_text(encoding="utf-8").splitlines()
            metrics = metrics_path.read_text(encoding="utf-8").splitlines()
            samples = samples_path.read_text(encoding="utf-8").splitlines()

        text = output.getvalue()
        self.assertIn("Character language-model comparison", text)
        self.assertIn("bigram", text)
        self.assertIn("embedding", text)
        self.assertIn("tiny-gpt", text)
        self.assertIn("train ppl", text)
        self.assertIn("dist-2", text)
        self.assertIn("bigram generated 1:", text)
        self.assertIn("tiny-gpt generated 2:", text)
        self.assertIn("summary file:", text)
        self.assertIn("metrics file:", text)
        self.assertIn("samples file:", text)
        self.assertEqual(
            summary[0],
            "model,model_type,context_size,parameters,train_loss,train_perplexity,accuracy,val_loss,val_perplexity,val_accuracy,sample_distinct_2,elapsed_seconds,examples_per_second,sample_count",
        )
        self.assertEqual(len(summary), 4)
        self.assertTrue(summary[1].startswith("bigram,bigram,4,"))
        self.assertEqual(
            metrics[0],
            "model,model_type,context_size,parameters,epoch,loss,perplexity,accuracy,val_loss,val_perplexity,val_accuracy,elapsed_seconds,examples_seen",
        )
        self.assertEqual(len(metrics), 4)
        self.assertTrue(metrics[1].startswith("bigram,bigram,4,"))
        self.assertEqual(
            samples[0],
            "model,model_type,context_size,parameters,sample_index,distinct_2,sample",
        )
        self.assertEqual(len(samples), 7)
        self.assertTrue(samples[1].startswith("bigram,bigram,4,"))


if __name__ == "__main__":
    unittest.main()
