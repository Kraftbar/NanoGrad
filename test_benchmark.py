import unittest

import importlib.util

from benchmark import CASES, average_seconds, case_data, nanograd_step, output_gradient, parse_args


class BenchmarkTests(unittest.TestCase):
    def test_parse_args(self) -> None:
        args = parse_args(["--repeat", "2", "--torch"])

        self.assertEqual(args.repeat, 2)
        self.assertTrue(args.torch)

    def test_average_seconds_runs_function(self) -> None:
        calls = []

        seconds = average_seconds(lambda: calls.append(1), repeat=3)

        self.assertEqual(calls, [1, 1, 1])
        self.assertGreaterEqual(seconds, 0.0)

    def test_average_seconds_rejects_invalid_repeat(self) -> None:
        with self.assertRaises(ValueError):
            average_seconds(lambda: None, repeat=0)


    def test_every_case_runs_forward_and_backward(self) -> None:
        for case in CASES:
            with self.subTest(case=case.name):
                inputs, targets = case_data(case)
                grad = output_gradient(case, inputs, targets)
                out, grads = nanograd_step(case, inputs, targets, grad)()
                self.assertEqual(len(out), len(grad))
                self.assertEqual([len(g) for g in grads], [len(x) for x in inputs])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_cases_agree_with_torch(self) -> None:
        from benchmark import _as_lists, max_difference, torch_step

        for case in CASES:
            with self.subTest(case=case.name):
                inputs, targets = case_data(case)
                grad = output_gradient(case, inputs, targets)
                ours = nanograd_step(case, inputs, targets, grad)()
                theirs = _as_lists(torch_step(case, inputs, targets, grad)())
                self.assertLess(max_difference([ours[0]], [theirs[0]]), 1e-9)
                self.assertLess(max_difference(ours[1], theirs[1]), 1e-9)


if __name__ == "__main__":
    unittest.main()
