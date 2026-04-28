import math
import unittest

from engine import Value


def finite_difference(
    fn,
    values: list[float],
    index: int,
    *,
    eps: float = 1e-6,
) -> float:
    plus = values[:]
    minus = values[:]

    plus[index] += eps
    minus[index] -= eps

    return (fn(plus) - fn(minus)) / (2 * eps)


class ValueTests(unittest.TestCase):
    def test_basic_backward(self) -> None:
        x = Value(2.0)
        y = Value(-3.0)
        z = x * y + x**2 + 4.0

        z.backward()

        self.assertAlmostEqual(z.data, 2.0)
        self.assertAlmostEqual(x.grad, 1.0)
        self.assertAlmostEqual(y.grad, 2.0)

    def test_tanh_backward_matches_derivative(self) -> None:
        x = Value(0.25)
        y = x.tanh()

        y.backward()

        expected = 1 - math.tanh(0.25) ** 2
        self.assertAlmostEqual(x.grad, expected)

    def test_backward_matches_finite_difference(self) -> None:
        def raw(values: list[float]) -> float:
            x, y, z = values
            return (
                x * y
                + z**2
                - x / 2.0
                + math.tanh(y - z)
            )

        def traced(values: list[float]) -> tuple[Value, list[Value]]:
            x, y, z = [Value(value) for value in values]
            out = (
                x * y
                + z**2
                - x / 2.0
                + (y - z).tanh()
            )
            return out, [x, y, z]

        values = [1.5, -0.75, 0.25]
        out, variables = traced(values)

        out.backward()

        for index, variable in enumerate(variables):
            numerical = finite_difference(raw, values, index)
            self.assertAlmostEqual(variable.grad, numerical, places=5)

    def test_relu_backward_matches_finite_difference_away_from_zero(self) -> None:
        def raw(values: list[float]) -> float:
            x, y = values
            return max(0.0, x * y - 0.25)

        def traced(values: list[float]) -> tuple[Value, list[Value]]:
            x, y = [Value(value) for value in values]
            out = (x * y - 0.25).relu()
            return out, [x, y]

        values = [2.0, 0.5]
        out, variables = traced(values)

        out.backward()

        for index, variable in enumerate(variables):
            numerical = finite_difference(raw, values, index)
            self.assertAlmostEqual(variable.grad, numerical, places=5)

    def test_exp_log_and_sigmoid_match_finite_difference(self) -> None:
        def raw(values: list[float]) -> float:
            x, y = values
            sigmoid = 1 / (1 + math.exp(-x))
            return math.exp(x * y) + math.log(y + 2.0) + sigmoid

        def traced(values: list[float]) -> tuple[Value, list[Value]]:
            x, y = [Value(value) for value in values]
            out = (x * y).exp() + (y + 2.0).log() + x.sigmoid()
            return out, [x, y]

        values = [0.4, 1.25]
        out, variables = traced(values)

        out.backward()

        for index, variable in enumerate(variables):
            numerical = finite_difference(raw, values, index)
            self.assertAlmostEqual(variable.grad, numerical, places=5)

    def test_log_rejects_non_positive_values(self) -> None:
        with self.assertRaises(ValueError):
            Value(0.0).log()


if __name__ == "__main__":
    unittest.main()
