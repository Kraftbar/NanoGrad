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

# (end of gemm tests)


# Convolutions, pooling, gathers and scatters.

# (end of conv tests)


# Rows and channels: softmax, norms, losses, GELU, the optimizer.

# (rows, width): one value; rows a warp takes (up to 1024 wide, 8 to a
# block: row counts that are not multiples of 8 leave spare warps) and rows
# a block takes (wider); widths that are not multiples of 32.
ROW_SHAPES = ((1, 1), (3, 5), (7, 37), (2, 300), (9, 1024), (5, 1031), (3, 2500), (130, 64))


class RowsTests(CudaTestCase):
    def test_gelu(self) -> None:
        for count in (1, 257, 3000):
            x, grad = values(count, 1, -5, 5), values(count, 2)
            with self.subTest(count=count):
                self.agree(lambda: kernels.gelu(x))
                self.agree(lambda: kernels.gelu_backward(x, kernels.gelu(x)[1], grad))

    def test_softmax_rows(self) -> None:
        for rows, width in ROW_SHAPES:
            x, grad = values(rows * width, 3, -20, 20), values(rows * width, 4)
            with self.subTest(rows=rows, width=width):
                self.agree(lambda: kernels.softmax_rows(x, width))
                self.agree(lambda: kernels.softmax_rows_backward(kernels.softmax_rows(x, width), grad, width))

    def test_softmax_xent_rows_works_in_place(self) -> None:
        for rows, width in ROW_SHAPES:
            x = values(rows * width, 5, -10, 10)
            targets = [random.Random(row).randrange(width) for row in range(rows)]
            if rows > 1:
                x[width + targets[1]] = -900.0                 # p_target underflows to 0: the logsumexp branch

            def run():
                logits = kernels.storage(x)
                grad, nll = kernels.softmax_xent_rows(logits, targets, 0.25, width)
                self.assertIs(grad, logits)
                return grad, nll

            with self.subTest(rows=rows, width=width):
                self.agree(run)

    def test_add_rows_adds_in_place(self) -> None:
        for rows, width in ROW_SHAPES + ((3000, 3),):
            x = values(rows * width, 6)

            def run():
                out = kernels.storage(values(width, 7))
                self.assertIs(kernels.add_rows(x, out, width), out)
                return out

            with self.subTest(rows=rows, width=width):
                self.agree(run)

    def test_layer_norm_rows(self) -> None:
        for rows, width in ROW_SHAPES + ((2000, 3),):
            x, grad = values(rows * width, 8, -3, 5), values(rows * width, 9)
            weight, bias = values(width, 10, 0.5, 2), values(width, 11)

            def backward(need_dx):
                _, x_hat, inv_std = kernels.layer_norm_rows(x, weight, bias, width, 1e-5)
                return kernels.layer_norm_rows_backward(grad, x_hat, inv_std, weight, width, need_dx)

            with self.subTest(rows=rows, width=width):
                self.agree(lambda: kernels.layer_norm_rows(x, weight, bias, width, 1e-5))
                self.agree(lambda: backward(True))
                self.agree(lambda: backward(False))

    def test_rms_norm_rows(self) -> None:
        for rows, width in ROW_SHAPES + ((2000, 3),):
            x, grad, weight = values(rows * width, 12, -3, 5), values(rows * width, 13), values(width, 14, 0.5, 2)

            def backward():
                _, inv_rms = kernels.rms_norm_rows(x, weight, width, 1e-6)
                return kernels.rms_norm_rows_backward(grad, x, inv_rms, weight, width)

            with self.subTest(rows=rows, width=width):
                self.agree(lambda: kernels.rms_norm_rows(x, weight, width, 1e-6))
                self.agree(backward)

    def test_batch_norm_channels(self) -> None:
        for batch, channels, plane in ((1, 1, 1), (4, 3, 1), (2, 5, 49), (3, 2, 700), (16, 4, 9)):
            size = batch * channels * plane
            x, grad = values(size, 15, -2, 4), values(size, 16)
            weight, bias = values(channels, 17, 0.5, 2), values(channels, 18)
            shape = {"batch": batch, "channels": channels, "plane": plane}

            def backward():
                _, x_hat, inv_std, _, _ = kernels.batch_norm_channels(x, weight, bias, eps=1e-5, **shape)
                return kernels.batch_norm_channels_backward(grad, x_hat, inv_std, weight, **shape)

            with self.subTest(**shape):
                self.agree(lambda: kernels.batch_norm_channels(x, weight, bias, eps=1e-5, **shape))
                self.agree(backward)

    def test_sums_are_floats(self) -> None:
        for count in (0, 1, 255, 100_000):
            x, t = values(count, 19, -30, 30), values(count, 20, 0, 1)
            with self.subTest(count=count):
                self.agree(lambda: kernels.sum_of_squares(x))
                self.agree(lambda: kernels.bce_with_logits_sum(x, t))
                self.agree(lambda: kernels.bce_with_logits_backward(x, t, 0.7))
        kernels.use_backend("cuda")
        self.assertIsInstance(kernels.sum_of_squares(kernels.storage([3.0, 4.0])), float)

    def test_dropout_mask_is_the_same_mask(self) -> None:
        for count, p, seed in ((1, 0.5, 0), (1000, 0.1, 12345), (4097, 0.5, 2**64 - 1), (300, 0.9, 2**63)):
            with self.subTest(count=count, seed=seed):
                c, cuda = self.run_both(lambda: kernels.dropout_mask(count, p, seed))
                self.assertEqual(c, cuda)

    def test_projected_sign_step(self) -> None:
        delta, grad, x = values(999, 21, -0.1, 0.1), values(999, 22), values(999, 23, 0, 1)
        grad[:10] = [0.0] * 10
        self.agree(lambda: kernels.projected_sign_step(delta, grad, x, 0.03, 0.05))

    def test_adam_updates_the_moments_in_place(self) -> None:
        w, grad = values(1001, 24), values(1001, 25)
        settings = {"grad_scale": 0.5, "beta1": 0.9, "beta2": 0.999, "decay": 0.99,
                    "step_size": 1e-3, "sqrt_correction": 0.3, "eps": 1e-8}

        def run():
            m, v = kernels.storage(values(1001, 26)), kernels.storage(values(1001, 27, 0, 1))
            new_w, new_m, new_v = kernels.adam_update(w, grad, m, v, **settings)
            self.assertIs(new_m, m)
            self.assertIs(new_v, v)
            return new_w, m, v

        self.agree(run)

    def test_float32(self) -> None:
        def float32(function):
            def run():
                kernels.set_default_dtype("float32")
                try:
                    return function()
                finally:
                    kernels.set_default_dtype("float64")
            return run

        x, grad, weight = values(9 * 300, 28, -3, 3), values(9 * 300, 29), values(300, 30, 0.5, 2)
        cases = {
            "gelu": lambda: kernels.gelu(kernels.storage(x)),
            "softmax": lambda: kernels.softmax_rows(kernels.storage(x), 300),
            "softmax_xent": lambda: kernels.softmax_xent_rows(kernels.storage(x), list(range(9)), 0.5, 300),
            "layer_norm": lambda: kernels.layer_norm_rows_backward(
                kernels.storage(grad), *kernels.layer_norm_rows(kernels.storage(x), weight, weight, 300, 1e-5)[1:],
                kernels.storage(weight), 300, True),
            "rms_norm": lambda: kernels.rms_norm_rows(kernels.storage(x), kernels.storage(weight), 300, 1e-6),
            "batch_norm": lambda: kernels.batch_norm_channels(kernels.storage(x), weight[:3], weight[:3],
                                                               batch=3, channels=3, plane=300, eps=1e-5),
            "sum_of_squares": lambda: kernels.sum_of_squares(kernels.storage(x)),
            "adam": lambda: kernels.adam_update(
                kernels.storage(x), kernels.storage(grad), kernels.zeros(len(x)), kernels.zeros(len(x)),
                grad_scale=1.0, beta1=0.9, beta2=0.999, decay=1.0, step_size=1e-3, sqrt_correction=1.0, eps=1e-8),
        }
        for name, case in cases.items():
            with self.subTest(kernel=name):
                self.agree(float32(case), tolerance=1e-5)

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
