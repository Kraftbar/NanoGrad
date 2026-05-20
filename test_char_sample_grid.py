import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from char_demo import save_checkpoint
from char_sample_grid import parse_args, parse_top_k, parse_temperature, run
from language import CharBigramModel
from text import CharVocab


class CharSampleGridTests(unittest.TestCase):
    def test_parse_args(self) -> None:
        args = parse_args([
            "--load-model",
            "char.json",
            "--output-dir",
            "grid",
            "--samples-file",
            "samples.csv",
            "--summary-file",
            "summary.csv",
            "--seed",
            "3",
            "--seed-text",
            "ab",
            "--generate",
            "4",
            "--num-samples",
            "2",
            "--sample-mode",
            "sample",
            "--sample-seed",
            "7",
            "--temperatures",
            "0.7",
            "1.0",
            "--top-k",
            "none",
            "2",
        ])

        self.assertEqual(args.load_model, Path("char.json"))
        self.assertEqual(args.output_dir, Path("grid"))
        self.assertEqual(args.samples_file, Path("samples.csv"))
        self.assertEqual(args.summary_file, Path("summary.csv"))
        self.assertEqual(args.seed, 3)
        self.assertEqual(args.seed_text, "ab")
        self.assertEqual(args.generate, 4)
        self.assertEqual(args.num_samples, 2)
        self.assertEqual(args.sample_mode, "sample")
        self.assertEqual(args.sample_seed, 7)
        self.assertEqual(args.temperatures, [0.7, 1.0])
        self.assertEqual(args.top_k, [None, 2])

    def test_parse_args_output_dir_sets_standard_files(self) -> None:
        args = parse_args([
            "--load-model",
            "char.json",
            "--output-dir",
            "grid",
        ])

        self.assertEqual(args.samples_file, Path("grid") / "samples.csv")
        self.assertEqual(args.summary_file, Path("grid") / "summary.csv")

    def test_parse_args_requires_samples_destination(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--load-model", "char.json"])

    def test_parse_grid_values_reject_invalid_values(self) -> None:
        self.assertIsNone(parse_top_k("none"))
        self.assertEqual(parse_top_k("4"), 4)
        with self.assertRaises(Exception):
            parse_top_k("0")
        with self.assertRaises(Exception):
            parse_temperature("0")

    def test_run_writes_sample_grid_csv(self) -> None:
        vocab = CharVocab.from_text("ab")
        model = CharBigramModel(len(vocab), context_size=1, seed=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "char.json"
            samples_path = Path(tmpdir) / "grid" / "samples.csv"
            summary_path = Path(tmpdir) / "grid" / "summary.csv"
            save_checkpoint(checkpoint_path, model, vocab, model_name="bigram")
            args = parse_args([
                "--load-model",
                str(checkpoint_path),
                "--samples-file",
                str(samples_path),
                "--summary-file",
                str(summary_path),
                "--seed-text",
                "a",
                "--generate",
                "2",
                "--num-samples",
                "2",
                "--sample-seed",
                "5",
                "--temperatures",
                "0.8",
                "1.0",
                "--top-k",
                "none",
                "1",
            ])
            output = io.StringIO()

            with redirect_stdout(output):
                run(args)

            rows = samples_path.read_text(encoding="utf-8").splitlines()
            summary = summary_path.read_text(encoding="utf-8").splitlines()

        text = output.getvalue()
        self.assertIn("Character sample grid", text)
        self.assertIn("summary file:", text)
        self.assertIn("rows:          8", text)
        self.assertEqual(
            rows[0],
            "checkpoint,model_type,context_size,parameters,temperature,top_k,sample_index,seed_text,sample_mode,sample_seed,distinct_2,sample",
        )
        self.assertEqual(len(rows), 9)
        self.assertIn(",bigram,1,", rows[1])
        self.assertIn(",a,sample,5,", rows[1])
        self.assertEqual(
            summary[0],
            "checkpoint,model_type,context_size,parameters,temperature,top_k,sample_mode,sample_seed,sample_count,mean_distinct_2",
        )
        self.assertEqual(len(summary), 5)

    def test_run_output_dir_writes_standard_files(self) -> None:
        vocab = CharVocab.from_text("ab")
        model = CharBigramModel(len(vocab), context_size=1, seed=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "char.json"
            output_dir = Path(tmpdir) / "grid"
            save_checkpoint(checkpoint_path, model, vocab, model_name="bigram")
            args = parse_args([
                "--load-model",
                str(checkpoint_path),
                "--output-dir",
                str(output_dir),
                "--seed-text",
                "a",
                "--generate",
                "1",
            ])
            output = io.StringIO()

            with redirect_stdout(output):
                run(args)

            samples = (output_dir / "samples.csv").read_text(encoding="utf-8")
            summary = (output_dir / "summary.csv").read_text(encoding="utf-8")

        text = output.getvalue()
        self.assertIn("output dir:", text)
        self.assertIn("samples file:", text)
        self.assertIn("summary file:", text)
        self.assertIn("sample_index", samples)
        self.assertIn("mean_distinct_2", summary)


if __name__ == "__main__":
    unittest.main()
