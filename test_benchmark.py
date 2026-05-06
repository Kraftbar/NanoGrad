import unittest

from benchmark import average_seconds, parse_args


class BenchmarkTests(unittest.TestCase):
    def test_parse_args(self) -> None:
        args = parse_args(["--repeat", "2"])

        self.assertEqual(args.repeat, 2)

    def test_average_seconds_runs_function(self) -> None:
        calls = []

        seconds = average_seconds(lambda: calls.append(1), repeat=3)

        self.assertEqual(calls, [1, 1, 1])
        self.assertGreaterEqual(seconds, 0.0)

    def test_average_seconds_rejects_invalid_repeat(self) -> None:
        with self.assertRaises(ValueError):
            average_seconds(lambda: None, repeat=0)


if __name__ == "__main__":
    unittest.main()
