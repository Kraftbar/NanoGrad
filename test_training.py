import random
import unittest

from datasets import line_fitting, sign_separator, xor_gate
from engine import Value
from metrics import binary_accuracy
from model import MLP
from train import binary_cross_entropy, train_binary_classifier, train_mse


class TrainingTests(unittest.TestCase):
    def test_tiny_regression_loss_decreases(self) -> None:
        random.seed(0)

        xs, ys = line_fitting()
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

    def test_binary_accuracy_counts_thresholded_predictions(self) -> None:
        accuracy = binary_accuracy(
            probabilities=[
                0.1,
                0.8,
                0.4,
                0.7,
            ],
            targets=[
                0.0,
                1.0,
                1.0,
                1.0,
            ],
        )

        self.assertEqual(accuracy, 0.75)

    def test_tiny_binary_classifier_loss_decreases(self) -> None:
        random.seed(0)

        xs, ys = sign_separator()
        model = MLP(inputs=1, layers=[1])
        history = train_binary_classifier(model, xs, ys, steps=60, lr=0.1)

        self.assertLess(history[-1], history[0])
        self.assertLess(history[-1], 0.2)

    def test_xor_classifier_learns_non_linear_pattern(self) -> None:
        random.seed(0)

        xs, ys = xor_gate()
        model = MLP(inputs=2, layers=[4, 1])
        history = train_binary_classifier(model, xs, ys, steps=1000, lr=0.2)

        probabilities = []
        for x in xs:
            logit = model(x)
            self.assertIsInstance(logit, Value)
            probabilities.append(logit.sigmoid().data)

        self.assertLess(history[-1], history[0])
        self.assertLess(history[-1], 0.2)
        self.assertEqual(binary_accuracy(probabilities, ys), 1.0)


if __name__ == "__main__":
    unittest.main()
