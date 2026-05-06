import unittest

from optim import TensorSGD
from tensor import Tensor


class OptimizerTests(unittest.TestCase):
    def test_tensor_sgd_updates_parameters(self) -> None:
        parameter = Tensor.from_list([1.0, -1.0], requires_grad=True)
        parameter.grad = [0.5, -0.25]

        TensorSGD([parameter], lr=0.2).step()

        self.assertEqual(parameter.data, [0.9, -0.95])

    def test_tensor_sgd_clips_global_gradient_norm(self) -> None:
        parameter = Tensor.from_list([1.0, 1.0], requires_grad=True)
        parameter.grad = [3.0, 4.0]

        TensorSGD([parameter], lr=1.0, max_grad_norm=1.0).step()

        self.assertAlmostEqual(parameter.data[0], 0.4)
        self.assertAlmostEqual(parameter.data[1], 0.2)

    def test_tensor_sgd_skips_clipping_small_gradient_norm(self) -> None:
        parameter = Tensor.from_list([1.0, 1.0], requires_grad=True)
        parameter.grad = [0.3, 0.4]

        TensorSGD([parameter], lr=1.0, max_grad_norm=1.0).step()

        self.assertEqual(parameter.data, [0.7, 0.6])

    def test_tensor_sgd_rejects_nonpositive_max_grad_norm(self) -> None:
        with self.assertRaises(ValueError):
            TensorSGD([], max_grad_norm=0.0)


if __name__ == "__main__":
    unittest.main()
