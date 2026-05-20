import io
import unittest
from contextlib import redirect_stdout

from language_benchmark import average_seconds, benchmark_transformer, parse_args, run


class LanguageBenchmarkTests(unittest.TestCase):
    def test_parse_args(self) -> None:
        args = parse_args([
            "--preset",
            "lite-gpt",
            "--batch-size",
            "2",
            "--repeat",
            "2",
        ])

        self.assertEqual(args.preset, "lite-gpt")
        self.assertEqual(args.batch_size, 2)
        self.assertEqual(args.repeat, 2)

    def test_average_seconds_runs_function(self) -> None:
        calls = []

        seconds = average_seconds(lambda: calls.append(1), repeat=3)

        self.assertEqual(calls, [1, 1, 1])
        self.assertGreaterEqual(seconds, 0.0)

    def test_average_seconds_rejects_invalid_repeat(self) -> None:
        with self.assertRaises(ValueError):
            average_seconds(lambda: None, repeat=0)

    def test_parse_args_rejects_invalid_batch_size(self) -> None:
        with self.assertRaises(ValueError):
            parse_args(["--batch-size", "0"])

    def test_benchmark_transformer_runs_one_pass(self) -> None:
        benchmark_transformer(
            {
                "context_size": 2,
                "embedding_dim": 4,
                "hidden_dim": 8,
                "heads": 2,
                "layers": 1,
                "activation": "gelu",
                "tie_embeddings": True,
            },
            batch_size=1,
        )

    def test_run_prints_language_benchmark(self) -> None:
        args = parse_args(["--repeat", "1"])
        output = io.StringIO()

        with redirect_stdout(output):
            run(args)

        text = output.getvalue()
        self.assertIn("NanoGrad language benchmarks", text)
        self.assertIn("preset: tiny-gpt", text)
        self.assertIn("batch size: 4", text)
        self.assertIn("tiny-gpt transformer fwd+bwd:", text)


if __name__ == "__main__":
    unittest.main()
