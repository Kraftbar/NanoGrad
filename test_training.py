import random
import unittest

from model import MLP
from train import train_mse


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


if __name__ == "__main__":
    unittest.main()
