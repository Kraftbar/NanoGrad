import io
import unittest
from contextlib import redirect_stdout

import tensor_demo


class TensorDemoTests(unittest.TestCase):
    def test_tensor_demo_runs(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            tensor_demo.main()

        text = output.getvalue()
        self.assertIn("Tensor regression loss", text)
        self.assertIn("mse:         2.666667", text)
        self.assertIn("Tensor binary classification", text)
        self.assertIn("accuracy:    0.750", text)
        self.assertIn("Tensor matrix-vector multiply", text)
        self.assertIn("output:  [140.0, 320.0]", text)
        self.assertIn("Tensor linear layer", text)
        self.assertIn("output: [[15.0, 31.0], [141.0, 319.0]]", text)
        self.assertIn("Tensor convolution and pooling", text)
        self.assertIn("features: [[-5.0, -5.0, -5.0]", text)
        self.assertIn("pooled:   [[-5.0]]", text)
        self.assertIn("Tensor conv training", text)
        self.assertIn("Tiny CNN classifier training", text)

    def test_conv_training_demo_learns_pattern(self) -> None:
        _layer, history = tensor_demo.train_conv_pattern()

        self.assertLess(history[-1], history[0])
        self.assertLess(history[-1], 0.05)

    def test_tiny_cnn_classifier_learns_synthetic_images(self) -> None:
        _conv, _classifier, history, accuracy = tensor_demo.train_tiny_cnn_classifier()

        self.assertLess(history[-1], history[0])
        self.assertLess(history[-1], 0.05)
        self.assertEqual(accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
