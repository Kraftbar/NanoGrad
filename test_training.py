import random
import unittest

from datasets import (
    and_gate,
    line_fitting,
    noisy_line_fitting,
    or_gate,
    sign_separator,
    tiny_2d_clusters,
    xor_gate,
)
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

    def test_noisy_regression_loss_decreases(self) -> None:
        random.seed(0)

        xs, ys = noisy_line_fitting()
        model = MLP(inputs=1, layers=[1])
        history = train_mse(model, xs, ys, steps=80, lr=0.03)

        self.assertLess(history[-1], history[0])
        self.assertLess(history[-1], 0.05)

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

    def test_tiny_2d_clusters_learn(self) -> None:
        random.seed(0)

        xs, ys = tiny_2d_clusters()
        model = MLP(inputs=2, layers=[1])
        history = train_binary_classifier(model, xs, ys, steps=80, lr=0.1)
        probabilities = binary_probabilities(model, xs)

        self.assertLess(history[-1], history[0])
        self.assertLess(history[-1], 0.1)
        self.assertEqual(binary_accuracy(probabilities, ys), 1.0)

    def test_linear_logic_gates_learn(self) -> None:
        for dataset in (and_gate, or_gate):
            with self.subTest(dataset=dataset.__name__):
                random.seed(0)

                xs, ys = dataset()
                model = MLP(inputs=2, layers=[1])
                history = train_binary_classifier(model, xs, ys, steps=300, lr=0.2)
                probabilities = binary_probabilities(model, xs)

                self.assertLess(history[-1], history[0])
                self.assertLess(history[-1], 0.2)
                self.assertEqual(binary_accuracy(probabilities, ys), 1.0)

    def test_xor_classifier_learns_non_linear_pattern(self) -> None:
        random.seed(0)

        xs, ys = xor_gate()
        model = MLP(inputs=2, layers=[4, 1])
        history = train_binary_classifier(model, xs, ys, steps=1000, lr=0.2)

        probabilities = binary_probabilities(model, xs)

        self.assertLess(history[-1], history[0])
        self.assertLess(history[-1], 0.2)
        self.assertEqual(binary_accuracy(probabilities, ys), 1.0)


def binary_probabilities(model: MLP, xs: list[list[float]]) -> list[float]:
    probabilities = []
    for x in xs:
        logit = model(x)
        if not isinstance(logit, Value):
            raise TypeError("expected the model to return one scalar Value")
        probabilities.append(logit.sigmoid().data)
    return probabilities


if __name__ == "__main__":
    unittest.main()
