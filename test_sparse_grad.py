import random
import unittest

from optim import TensorAdam, TensorSGD
from tensor import RowGrad, Tensor, gather_rows, matmul


def train(sparse: bool, optimizer_type, steps: int = 3, dense_use: bool = False) -> list[float]:
    rng = random.Random(0)
    table = Tensor([rng.uniform(-1, 1) for _ in range(40)], (8, 5), requires_grad=True)
    table.sparse_grad = sparse
    optimizer = optimizer_type([table], lr=0.3, max_grad_norm=0.5)
    for _ in range(steps):
        optimizer.zero_grad()
        picked = gather_rows(table, [1, 6, 1])
        loss = (picked * Tensor([rng.uniform(-1, 1) for _ in range(15)], (3, 5))).sum()
        if dense_use:                                    # the table also used by an ordinary op
            loss = loss + matmul(table, Tensor([0.1] * 5, (5, 1))).sum()
        loss.backward()
        optimizer.step()
    return list(table.data)


class SparseGradTests(unittest.TestCase):
    def test_sgd_with_row_gradients_equals_dense(self) -> None:
        for dense_use in (False, True):
            with self.subTest(dense_use=dense_use):
                for a, b in zip(train(True, TensorSGD, dense_use=dense_use), train(False, TensorSGD, dense_use=dense_use)):
                    self.assertAlmostEqual(a, b, places=14)

    def test_adam_accepts_row_gradients(self) -> None:
        for a, b in zip(train(True, TensorAdam), train(False, TensorAdam)):
            self.assertAlmostEqual(a, b, places=14)

    def test_row_gradient_keeps_only_used_rows(self) -> None:
        table = Tensor([1.0] * 12, (4, 3), requires_grad=True)
        table.sparse_grad = True
        gather_rows(table, [2, 0, 2]).sum().backward()
        self.assertIsInstance(table.grad, RowGrad)
        merged = table.grad.coalesced()
        self.assertEqual(sorted(merged.ids), [0, 2])
        self.assertEqual(list(table.grad.to_dense()), [1.0] * 3 + [0.0] * 3 + [2.0] * 3 + [0.0] * 3)


if __name__ == "__main__":
    unittest.main()
