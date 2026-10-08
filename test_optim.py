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


class AdamL2Tests(unittest.TestCase):
    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "PyTorch not installed")
    def test_l2_matches_torch_adam_weight_decay(self) -> None:
        import torch

        from optim import TensorAdam
        from tensor import Tensor

        values = [0.5, -1.0, 2.0]
        ours = Tensor(list(values), (3,), requires_grad=True)
        theirs = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        optimizer = TensorAdam([ours], lr=0.01, l2=0.05)
        reference = torch.optim.Adam([theirs], lr=0.01, weight_decay=0.05)
        for step in range(5):
            grad = [0.1 * (step + 1), -0.2, 0.3]
            ours.grad = list(grad)
            theirs.grad = torch.tensor(grad, dtype=torch.float64)
            optimizer.step()
            reference.step()
        for a, b in zip(ours.data, theirs.tolist()):
            self.assertAlmostEqual(a, b, places=14)


class LBFGSTests(unittest.TestCase):
    def test_minimizes_a_quadratic_quickly(self) -> None:
        from optim import TensorLBFGS
        from tensor import Tensor

        x = Tensor([3.0, -2.0], (2,), requires_grad=True)
        target = Tensor([1.0, 4.0], (2,))
        optimizer = TensorLBFGS([x], lr=1.0)

        def closure() -> float:
            x.zero_grad()
            loss = ((x - target) * (x - target) * Tensor([1.0, 10.0], (2,))).sum()
            loss.backward()
            return loss.item()

        optimizer.step(closure)
        for a, b in zip(x.data, [1.0, 4.0]):
            self.assertAlmostEqual(a, b, places=6)

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "PyTorch not installed")
    def test_takes_the_same_steps_as_torch(self) -> None:
        import random

        import torch

        from optim import TensorLBFGS
        from tensor import Tensor, matmul

        rng = random.Random(0)
        xs = [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(20)]
        ys = [sum(row) ** 2 + 0.3 for row in xs]
        w1, w2 = [rng.uniform(-0.5, 0.5) for _ in range(24)], [rng.uniform(-0.5, 0.5) for _ in range(8)]
        ours = [Tensor(w1, (3, 8), requires_grad=True), Tensor(w2, (8, 1), requires_grad=True)]
        theirs = [torch.tensor(w1, dtype=torch.float64).reshape(3, 8).requires_grad_(),
                  torch.tensor(w2, dtype=torch.float64).reshape(8, 1).requires_grad_()]
        x_ours, y_ours = Tensor([v for row in xs for v in row], (20, 3)), Tensor(ys, (20, 1))
        x_torch, y_torch = torch.tensor(xs, dtype=torch.float64), torch.tensor(ys, dtype=torch.float64).reshape(20, 1)
        optimizer_ours, optimizer_torch = TensorLBFGS(ours, lr=0.8), torch.optim.LBFGS(theirs, lr=0.8)

        def closure_ours() -> float:
            for parameter in ours:
                parameter.zero_grad()
            error = matmul(matmul(x_ours, ours[0]).tanh(), ours[1]) - y_ours
            loss = (error * error).mean()
            loss.backward()
            return loss.item()

        def closure_torch():
            optimizer_torch.zero_grad()
            loss = ((torch.tanh(x_torch @ theirs[0]) @ theirs[1] - y_torch) ** 2).mean()
            loss.backward()
            return loss

        for _ in range(2):
            self.assertAlmostEqual(optimizer_ours.step(closure_ours), optimizer_torch.step(closure_torch).item(), places=10)
        for parameter, reference in zip(ours, theirs):
            for a, b in zip(parameter.data, reference.detach().reshape(-1).tolist()):
                self.assertAlmostEqual(a, b, places=9)


if __name__ == "__main__":
    unittest.main()
