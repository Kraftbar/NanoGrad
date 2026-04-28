import random
import unittest

from engine import Value
from model import MLP
from train import binary_cross_entropy, train_binary_classifier, train_mse


class TrainingTests(unittest.TestCase):
    def test_tiny_regression_loss_decreases(self) -> None:
        random.seed(0)

        xs = [
            [-1.0],
            [0.0],
            [1.0],
            [2.0],
        ]
        ys = [
            -2.0,
            1.0,
            4.0,
            7.0,
        ]

        model = MLP(inputs=1, layers=[1])
        history = train_mse(model, xs, ys, steps=40, lr=0.05)

        self.assertLess(history[-1], history[0])
        self.assertLess(history[-1], 0.01)

    def test_binary_cross_entropy_prefers_correct_confident_outputs(self) -> None:
        good = binary_cross_entropy(
            probabilities=[
                Value(0.01),
                Value(0.99),
            ],
            targets=[
                0.0,
                1.0,
            ],
        )
        bad = binary_cross_entropy(
            probabilities=[
                Value(0.99),
                Value(0.01),
            ],
            targets=[
                0.0,
                1.0,
            ],
        )

        self.assertLess(good.data, bad.data)

    def test_tiny_binary_classifier_loss_decreases(self) -> None:
        random.seed(0)

        xs = [
            [-2.0],
            [-1.0],
            [1.0],
            [2.0],
        ]
        ys = [
            0.0,
            0.0,
            1.0,
            1.0,
        ]

        model = MLP(inputs=1, layers=[1])
        history = train_binary_classifier(model, xs, ys, steps=60, lr=0.1)

        self.assertLess(history[-1], history[0])
        self.assertLess(history[-1], 0.2)


if __name__ == "__main__":
    unittest.main()
