"""The CUDA kernels against the C ones (skipped without an NVIDIA GPU)."""

import pickle
import random
import unittest
from array import array

import kernels
import kernels_cuda
from tensor import Tensor

CUDA = kernels_cuda.load()


def values(count, seed, low=-1.0, high=1.0):
    rng = random.Random(seed)
    return [rng.uniform(low, high) for _ in range(count)]


def flat(result):
    """Kernel results as plain lists of floats (tuples of arrays stay tuples)."""

    if isinstance(result, tuple):
        return tuple(flat(r) for r in result)
    if isinstance(result, (int, float)) or result is None:
        return result
    return [float(v) for v in result]


@unittest.skipUnless(CUDA and kernels.BACKEND in ("c", "cuda"), "needs an NVIDIA GPU and the C kernels")
class CudaTestCase(unittest.TestCase):
    """Run the same function on the C and the CUDA backend (float64) and compare."""

    def setUp(self) -> None:
        self.previous = kernels.BACKEND

    def tearDown(self) -> None:
        kernels.use_backend(self.previous)

    def run_both(self, function):
        results = []
        for backend in ("c", "cuda"):
            kernels.use_backend(backend)
            results.append(flat(function()))
        return results

    def assertClose(self, c, cuda, tolerance=1e-12) -> None:
        if isinstance(c, tuple):
            self.assertEqual(len(c), len(cuda))
            for left, right in zip(c, cuda):
                self.assertClose(left, right, tolerance)
            return
        if isinstance(c, (int, float)) or c is None:
            self.assertAlmostEqual(c, cuda, delta=tolerance * max(1.0, abs(c or 0.0)))
            return
        self.assertEqual(len(c), len(cuda))
        for index, (a, b) in enumerate(zip(c, cuda)):
            if abs(a - b) > tolerance * max(1.0, abs(a)):
                self.fail(f"value {index}: C {a!r}, CUDA {b!r}")

    def agree(self, function, tolerance=1e-12) -> None:
        c, cuda = self.run_both(function)
        self.assertClose(c, cuda, tolerance)


class DeviceArrayTests(CudaTestCase):
    def setUp(self) -> None:
        super().setUp()
        kernels.use_backend("cuda")

    def test_reads_like_an_array(self) -> None:
        data = kernels.storage([1.0, 2.0, 3.0, 4.0])
        self.assertIsInstance(data, kernels_cuda.DeviceArray)
        self.assertEqual(len(data), 4)
        self.assertEqual(data[2], 3.0)
        self.assertEqual(list(data), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(sum(data), 10.0)
        self.assertEqual(list(data[1:3]), [2.0, 3.0])
        self.assertEqual(list(data * 2), [1.0, 2.0, 3.0, 4.0] * 2)
        self.assertEqual(list(data[:2] + data[3:]), [1.0, 2.0, 4.0])
        self.assertEqual(list(kernels_cuda.DeviceArray("d", [5.0, 6.0])), [5.0, 6.0])

    def test_host_copy_is_made_once_and_dropped_when_a_kernel_writes(self) -> None:
        data = kernels.storage([1.0, 2.0])
        before = kernels_cuda.syncs
        data[0], data[1], list(data)
        self.assertEqual(kernels_cuda.syncs, before + 1)
        data[1:2] = kernels.storage([7.0])
        self.assertEqual(list(data), [1.0, 7.0])
        data[0] = 5.0
        self.assertEqual(list(data), [5.0, 7.0])

    def test_pickles_with_its_values(self) -> None:
        data = kernels.storage([1.5, -2.0])
        self.assertEqual(list(pickle.loads(pickle.dumps(data))), [1.5, -2.0])

    def test_zeros_and_index_tables(self) -> None:
        self.assertEqual(list(kernels.zeros(3)), [0.0, 0.0, 0.0])
        table = kernels.index_table([2, 0, 1])
        self.assertEqual(table.typecode, "l")
        self.assertEqual(list(kernels.gather_indexed(kernels.storage([10.0, 20.0, 30.0]), table,
                                                     batches=1, image_size=3)), [30.0, 10.0, 20.0])

    def test_float32(self) -> None:
        kernels.set_default_dtype("float32")
        try:
            data = kernels.storage([0.1, 0.2])
            self.assertEqual(data.typecode, "f")
            self.assertEqual(list(kernels.add(data, data)), list(array("f", [0.2, 0.4])))
        finally:
            kernels.set_default_dtype("float64")


class ElementwiseTests(CudaTestCase):
    def test_elementwise_kernels_agree(self) -> None:
        x, y, positive = values(1000, 1, -4, 4), values(1000, 2), values(1000, 3, 0.1, 5)
        cases = {
            "add": (x, y), "multiply": (x, y), "scale": (x, 0.37), "add_scaled": (x, y, -1.5),
            "reciprocal": (positive,), "exp": (x,), "log": (positive,), "sqrt": (positive,),
            "relu": (x,), "relu_backward": (x, y), "tanh": (x,), "tanh_backward": (y, x),
            "sigmoid": (x,), "sigmoid_backward": (y, x),
        }
        for name, args in cases.items():
            with self.subTest(kernel=name):
                self.agree(lambda: getattr(kernels, name)(*args))


# Matrix products (gemm).

class GemmTests(CudaTestCase):
    # Odd sizes around the tiles (32, 64, 128) and the k steps (8, 16),
    # matrix-vector shapes (the dot-product kernel), and a long k (split-k).
    SHAPES = ((1, 1, 1), (3, 4, 5), (1, 37, 9), (9, 37, 1), (17, 33, 65), (130, 9, 129), (40, 1500, 3))

    def gemm(self, m, k, n, batches, steps, trans_a, trans_b, initial=False):
        a, b = values(m * k * batches, m + k), values(k * n * batches, k + n)
        size = steps[2] * (batches - 1) + m * n if steps[2] else m * n
        start = values(size, 7) if initial else None
        return lambda: kernels.gemm(a, b, batches=batches, m=m, k=k, n=n, a_step=steps[0], b_step=steps[1],
                                    c_step=steps[2], trans_a=trans_a, trans_b=trans_b,
                                    initial=None if start is None else kernels.storage(start))

    def cases(self):
        for m, k, n in self.SHAPES:
            for trans_a in (False, True):
                for trans_b in (False, True):
                    for batches, steps in ((1, (0, 0, 0)), (3, (m * k, k * n, m * n)), (2, (0, k * n, m * n + 2)),
                                           (3, (m * k, 0, 0)), (3, (m * k, k * n, 0))):
                        yield m, k, n, batches, steps, trans_a, trans_b

    def test_every_layout_and_batch_step_agrees(self) -> None:
        for case in self.cases():
            for initial in (False, True):
                with self.subTest(case=case, initial=initial):
                    self.agree(self.gemm(*case, initial=initial))

    def test_every_tile_size_agrees(self) -> None:
        # float64 normally uses only the 32 x 32 tile; force each one in turn.
        saved = kernels_cuda._GEMM_TILES["d"]
        try:
            for edge in (128, 64):
                kernels_cuda._GEMM_TILES["d"] = (edge,)
                for m, k, n in ((130, 70, 260), (64, 16, 64), (5, 3000, 7)):
                    for trans_a in (False, True):
                        for trans_b in (False, True):
                            for batches, steps in ((1, (0, 0, 0)), (2, (m * k, k * n, m * n)), (3, (m * k, 0, 0))):
                                with self.subTest(edge=edge, shape=(m, k, n), layout=(trans_a, trans_b),
                                                  batches=batches):
                                    self.agree(self.gemm(m, k, n, batches, steps, trans_a, trans_b, initial=True))
        finally:
            kernels_cuda._GEMM_TILES["d"] = saved

    def test_float32_agrees(self) -> None:
        import kernels_c

        try:
            kernels_c.set_precision("f")
            kernels_cuda.set_precision("f")
            for m, k, n in ((300, 200, 270), (100, 784, 256), (70, 40, 1)):
                for trans_a in (False, True):
                    for trans_b in (False, True):
                        a, b = values(m * k, 1), values(k * n, 2)
                        args = (1, m, k, n, 0, 0, 0, trans_a, trans_b)
                        c = list(kernels_c.gemm(array("f", a), array("f", b), *args))
                        cuda = kernels_cuda.gemm(a, b, *args)
                        self.assertEqual(cuda.typecode, "f")
                        self.assertClose(c, list(cuda), tolerance=1e-4)
        finally:
            kernels_c.set_precision("d")
            kernels_cuda.set_precision("d")

    def test_initial_is_consumed_in_place(self) -> None:
        kernels.use_backend("cuda")
        start = kernels.storage([1.0, 2.0, 3.0, 4.0])
        list(start)                                   # a host copy, which the product must drop
        out = kernels.gemm([1.0, 2.0], [3.0, 4.0], batches=1, m=2, k=1, n=2, a_step=0, b_step=0, c_step=0,
                           initial=start)
        self.assertIs(out, start)
        self.assertEqual(list(out), [4.0, 6.0, 9.0, 12.0])
        with self.assertRaises(ValueError):
            kernels.gemm([1.0], [1.0], batches=1, m=1, k=1, n=1, a_step=0, b_step=0, c_step=0,
                         initial=kernels.storage([0.0, 0.0]))

    def test_matmul_and_linear_agree(self) -> None:
        from tensor import matmul
        from tensor_nn import TensorLinear

        cases = [((4,), (4, 3)), ((5, 4), (4,)), ((2, 3, 4), (4, 5)), ((2, 3, 4), (2, 4, 5)), ((3, 4), (2, 4, 5))]
        for left_shape, right_shape in cases:
            with self.subTest(left=left_shape, right=right_shape):
                def run():
                    left = Tensor(values(_gemm_numel(left_shape), 5), left_shape, requires_grad=True)
                    right = Tensor(values(_gemm_numel(right_shape), 6), right_shape, requires_grad=True)
                    out = matmul(left, right)
                    (out * Tensor(values(out.numel, 7), out.shape)).sum().backward()
                    return list(out.data), list(left.grad), list(right.grad)
                self.agree(run)

        def linear():
            layer = TensorLinear(20, 7, seed=3)
            x = Tensor(values(6 * 20, 8), (6, 20), requires_grad=True)
            (layer(x) * Tensor(values(6 * 7, 9), (6, 7))).sum().backward()
            return tuple(list(t) for t in (x.grad, layer.weight.grad, layer.bias.grad))
        self.agree(linear)


def _gemm_numel(shape):
    count = 1
    for size in shape:
        count *= size
    return count

# (end of gemm tests)


# Convolutions, pooling, gathers and scatters.

# (end of conv tests)


# Rows and channels: softmax, norms, losses, GELU, the optimizer.

# (end of rows tests)


# Recurrent: the LSTM cell and layer, the CRF.

# (end of recurrent tests)


class ModelTests(CudaTestCase):
    def test_a_small_network_trains_the_same(self) -> None:
        from optim import TensorAdam
        from losses import softmax_cross_entropy
        from tensor_nn import TensorConv2D, TensorLinear

        def train():
            conv = TensorConv2D((4, 1, 3, 3), padding=1, seed=0)
            linear = TensorLinear(4 * 4 * 4, 3, seed=1)
            parameters = conv.parameters() + linear.parameters()
            optimizer = TensorAdam(parameters, lr=1e-2)
            images = Tensor(values(2 * 64, 4), (2, 1, 8, 8))
            for _ in range(5):
                optimizer.zero_grad()
                hidden = conv(images).relu().max_pool2d((2, 2)).flatten(start_axis=1)
                softmax_cross_entropy(linear(hidden), [0, 2]).backward()
                optimizer.step()
            return tuple(list(p.data) for p in parameters)

        self.agree(train, tolerance=1e-10)


if __name__ == "__main__":
    unittest.main()
