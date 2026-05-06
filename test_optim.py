import unittest

from optim import TensorAdam, TensorSGD
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

    def test_tensor_sgd_applies_weight_decay(self) -> None:
        parameter = Tensor.from_list([1.0], requires_grad=True)
        parameter.grad = [0.0]

        TensorSGD([parameter], lr=0.1, weight_decay=0.5).step()

        self.assertAlmostEqual(parameter.data[0], 0.95)

    def test_tensor_sgd_rejects_invalid_options(self) -> None:
        with self.assertRaises(ValueError):
            TensorSGD([], max_grad_norm=0.0)

        with self.assertRaises(ValueError):
            TensorSGD([], weight_decay=-0.1)

    def test_tensor_adam_updates_parameters(self) -> None:
        parameter = Tensor.from_list([1.0, -1.0], requires_grad=True)
        optimizer = TensorAdam([parameter], lr=0.1)

        parameter.grad = [1.0, -1.0]
        optimizer.step()
        parameter.grad = [1.0, -1.0]
        optimizer.step()

        self.assertEqual(optimizer.step_count, 2)
        self.assertAlmostEqual(parameter.data[0], 0.8)
        self.assertAlmostEqual(parameter.data[1], -0.8)

    def test_tensor_adam_zero_grad(self) -> None:
        parameter = Tensor.from_list([1.0], requires_grad=True)
        parameter.grad = [2.0]

        TensorAdam([parameter]).zero_grad()

        self.assertEqual(parameter.grad, [0.0])

    def test_tensor_adam_applies_decoupled_weight_decay(self) -> None:
        parameter = Tensor.from_list([1.0], requires_grad=True)
        parameter.grad = [0.0]

        TensorAdam([parameter], lr=0.1, weight_decay=0.5).step()

        self.assertAlmostEqual(parameter.data[0], 0.95)

    def test_tensor_adam_rejects_invalid_hyperparameters(self) -> None:
        with self.assertRaises(ValueError):
            TensorAdam([], lr=0.0)

        with self.assertRaises(ValueError):
            TensorAdam([], beta1=1.0)

        with self.assertRaises(ValueError):
            TensorAdam([], beta2=1.0)

        with self.assertRaises(ValueError):
            TensorAdam([], eps=0.0)

        with self.assertRaises(ValueError):
            TensorAdam([], weight_decay=-0.1)

        with self.assertRaises(ValueError):
            TensorAdam([], max_grad_norm=0.0)


if __name__ == "__main__":
    unittest.main()
