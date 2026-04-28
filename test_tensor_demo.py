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


if __name__ == "__main__":
    unittest.main()
