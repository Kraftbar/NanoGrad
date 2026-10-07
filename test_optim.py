import unittest

from optim import TensorAdam, TensorSGD
from tensor import Tensor


class OptimizerTests(unittest.TestCase):
    def test_tensor_sgd_updates_parameters(self) -> None:
        parameter = Tensor.from_list([1.0, -1.0], requires_grad=True)
        parameter.grad = [0.5, -0.25]

        TensorSGD([parameter], lr=0.2).step()

        self.assertEqual(list(parameter.data), [0.9, -0.95])

    def test_tensor_sgd_clips_global_gradient_norm(self) -> None:
        parameter = Tensor.from_list([1.0, 1.0], requires_grad=True)
        parameter.grad = [3.0, 4.0]

        TensorSGD([parameter], lr=1.0, max_grad_norm=1.0).step()

        self.assertAlmostEqual(parameter.data[0], 0.4, places=6)        # scale 1 / (5 + 1e-6), as in PyTorch
        self.assertAlmostEqual(parameter.data[1], 0.2, places=6)

    def test_tensor_sgd_skips_clipping_small_gradient_norm(self) -> None:
        parameter = Tensor.from_list([1.0, 1.0], requires_grad=True)
        parameter.grad = [0.3, 0.4]

        TensorSGD([parameter], lr=1.0, max_grad_norm=1.0).step()

        self.assertEqual(list(parameter.data), [0.7, 0.6])

    def test_tensor_sgd_applies_weight_decay(self) -> None:
        parameter = Tensor.from_list([1.0], requires_grad=True)
        parameter.grad = [0.0]

        TensorSGD([parameter], lr=0.1, weight_decay=0.5).step()

        self.assertAlmostEqual(parameter.data[0], 0.95)

    def test_tensor_sgd_can_decay_only_higher_rank_parameters(self) -> None:
        vector = Tensor.from_list([1.0], requires_grad=True)
        matrix = Tensor.from_list([[1.0]], requires_grad=True)
        vector.grad = [0.0]
        matrix.grad = [0.0]

        TensorSGD(
            [vector, matrix],
            lr=0.1,
            weight_decay=0.5,
            weight_decay_min_ndim=2,
        ).step()

        self.assertAlmostEqual(vector.data[0], 1.0)
        self.assertAlmostEqual(matrix.data[0], 0.95)

    def test_tensor_sgd_rejects_invalid_options(self) -> None:
        with self.assertRaises(ValueError):
            TensorSGD([], max_grad_norm=0.0)

        with self.assertRaises(ValueError):
            TensorSGD([], weight_decay=-0.1)

        with self.assertRaises(ValueError):
            TensorSGD([], weight_decay_min_ndim=0)

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

        self.assertEqual(list(parameter.grad), [0.0])

    def test_tensor_adam_applies_decoupled_weight_decay(self) -> None:
        parameter = Tensor.from_list([1.0], requires_grad=True)
        parameter.grad = [0.0]

        TensorAdam([parameter], lr=0.1, weight_decay=0.5).step()

        self.assertAlmostEqual(parameter.data[0], 0.95)

    def test_tensor_adam_can_decay_only_higher_rank_parameters(self) -> None:
        vector = Tensor.from_list([1.0], requires_grad=True)
        matrix = Tensor.from_list([[1.0]], requires_grad=True)
        vector.grad = [0.0]
        matrix.grad = [0.0]

        TensorAdam(
            [vector, matrix],
            lr=0.1,
            weight_decay=0.5,
            weight_decay_min_ndim=2,
        ).step()

        self.assertAlmostEqual(vector.data[0], 1.0)
        self.assertAlmostEqual(matrix.data[0], 0.95)

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
            TensorAdam([], weight_decay_min_ndim=0)

        with self.assertRaises(ValueError):
            TensorAdam([], max_grad_norm=0.0)



class MomentumTests(unittest.TestCase):
    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "PyTorch is not installed")
    def test_momentum_and_nesterov_match_pytorch(self) -> None:
        import random

        import torch

        from optim import TensorSGD
        from tensor import Tensor

        for nesterov in (False, True):
            with self.subTest(nesterov=nesterov):
                rng = random.Random(1)
                start = [rng.uniform(-1.0, 1.0) for _ in range(6)]
                ours = Tensor(start, (2, 3), requires_grad=True)
                theirs = torch.tensor(start, dtype=torch.float64).reshape(2, 3).requires_grad_()
                optimizer = TensorSGD([ours], lr=0.1, momentum=0.9, nesterov=nesterov, weight_decay=0.01)
                reference = torch.optim.SGD([theirs], lr=0.1, momentum=0.9, nesterov=nesterov, weight_decay=0.01)
                for step in range(4):
                    target = [rng.uniform(-1.0, 1.0) for _ in range(6)]
                    optimizer.zero_grad()
                    ((ours - Tensor(target, (2, 3))) * (ours - Tensor(target, (2, 3)))).sum().backward()
                    optimizer.step()
                    reference.zero_grad()
                    ((theirs - torch.tensor(target, dtype=torch.float64).reshape(2, 3)) ** 2).sum().backward()
                    reference.step()
                for a, b in zip(ours.data, theirs.detach().reshape(-1).tolist()):
                    self.assertAlmostEqual(a, b, places=12)


if __name__ == "__main__":
    unittest.main()
