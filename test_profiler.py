import unittest

from profiler import profile
from tensor import Tensor


class ProfilerTests(unittest.TestCase):
    def test_counts_ops_and_times_backward(self) -> None:
        x = Tensor([1.0, 2.0, 3.0], (3,), requires_grad=True)
        with profile() as timings:
            y = (x * x).sum()
            y.backward()
        self.assertEqual(timings.calls["*"], 1)
        self.assertIn("*", timings.backward_seconds)
        self.assertIn("sum", timings.table())
        self.assertEqual(list(x.grad), [2.0, 4.0, 6.0])

    def test_profiles_do_not_nest(self) -> None:
        with profile():
            with self.assertRaises(RuntimeError):
                with profile():
                    pass


if __name__ == "__main__":
    unittest.main()
