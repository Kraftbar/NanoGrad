import random
import unittest

import kernels
from tensor import Tensor, matmul


def values(count, seed):
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(count)]


@unittest.skipUnless(kernels.BACKEND == "c", "C kernels are not built here")
class KernelBackendTests(unittest.TestCase):
    def tearDown(self) -> None:
        kernels.use_backend("c")

    def run_both(self, function):
        results = []
        for backend in ("python", "c"):
            kernels.use_backend(backend)
            results.append(function())
        return results

    def assertListsClose(self, left, right) -> None:
        self.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            self.assertAlmostEqual(a, b, places=10)

    def test_gemm_backends_agree_for_every_layout(self) -> None:
        m, k, n, batches = 3, 4, 5, 2
        for trans_a in (False, True):
            for trans_b in (False, True):
                for a_step, b_step, c_step in ((m * k, k * n, m * n), (0, k * n, m * n), (m * k, 0, 0)):
                    with self.subTest(trans_a=trans_a, trans_b=trans_b, steps=(a_step, b_step, c_step)):
                        a = values(m * k * batches, 1)
                        b = values(k * n * batches, 2)
                        python, c = self.run_both(lambda: kernels.gemm(
                            a, b, batches=batches, m=m, k=k, n=n,
                            a_step=a_step, b_step=b_step, c_step=c_step,
                            trans_a=trans_a, trans_b=trans_b,
                        ))
                        self.assertListsClose(python, c)

    def test_step_zero_output_sums_the_batches(self) -> None:
        a = values(2 * 2 * 3, 3)
        b = values(3 * 2, 4)

        summed = kernels.gemm(a, b, batches=2, m=2, k=3, n=2, a_step=6, b_step=0, c_step=0)
        first = kernels.gemm(a[:6], b, batches=1, m=2, k=3, n=2, a_step=0, b_step=0, c_step=0)
        second = kernels.gemm(a[6:], b, batches=1, m=2, k=3, n=2, a_step=0, b_step=0, c_step=0)

        self.assertListsClose(summed, [x + y for x, y in zip(first, second)])

    def test_matmul_forward_and_backward_agree(self) -> None:
        cases = [((4,), (4, 3)), ((2, 3, 4), (4, 5)), ((2, 3, 4), (2, 4, 5)), ((3, 4), (2, 4, 5))]
        for left_shape, right_shape in cases:
            with self.subTest(left=left_shape, right=right_shape):
                def run():
                    left = Tensor(values(_numel(left_shape), 5), left_shape, requires_grad=True)
                    right = Tensor(values(_numel(right_shape), 6), right_shape, requires_grad=True)
                    out = matmul(left, right)
                    weights = Tensor(values(out.numel, 7), out.shape)
                    (out * weights).sum().backward()
                    return out.data, left.grad, right.grad
                python, c = self.run_both(run)
                for p, q in zip(python, c):
                    self.assertListsClose(p, q)

    def test_unknown_backend(self) -> None:
        with self.assertRaises(ValueError):
            kernels.use_backend("cuda")


def _numel(shape):
    total = 1
    for dim in shape:
        total *= dim
    return total


if __name__ == "__main__":
    unittest.main()
