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


def patch_index(channels, rows, cols, kernel_rows, kernel_cols, stride=1):
    """tensor._conv2d_patch_index, uncached (a table for the active backend)."""

    offsets = [c * rows * cols + i * cols + j
               for c in range(channels) for i in range(kernel_rows) for j in range(kernel_cols)]
    return kernels.index_table(
        y * stride * cols + x * stride + offset
        for y in range((rows - kernel_rows) // stride + 1)
        for x in range((cols - kernel_cols) // stride + 1)
        for offset in offsets)


def window_index(rows, cols, window_rows, window_cols, stride):
    return kernels.index_table(
        (y * stride + i) * cols + x * stride + j
        for y in range((rows - window_rows) // stride + 1)
        for x in range((cols - window_cols) // stride + 1)
        for i in range(window_rows) for j in range(window_cols))


class ConvTests(CudaTestCase):
    # (batches, channels, rows, cols, filters, kernel rows, kernel cols, stride): one value; odd
    # sizes; more than one tile of filters; a patch size not a multiple of 16; one large image
    # (kernel_grad splits its positions); many small images with many tiles (runs of images).
    SHAPES = [(1, 1, 1, 1, 1, 1, 1, 1), (3, 3, 9, 7, 5, 3, 3, 1), (2, 5, 13, 11, 70, 3, 2, 2),
              (4, 8, 20, 20, 17, 5, 5, 1), (1, 2, 40, 40, 3, 3, 3, 1), (40, 30, 5, 5, 130, 3, 3, 1)]

    def test_convolution_products_agree(self) -> None:
        for shape in self.SHAPES:
            batches, channels, rows, cols, filters, kh, kw, stride = shape
            image = values(batches * channels * rows * cols, 1)
            kernel = values(filters * channels * kh * kw, 2)
            positions = ((rows - kh) // stride + 1) * ((cols - kw) // stride + 1)
            grad = values(batches * filters * positions, 3)
            sizes = dict(batches=batches, image_size=channels * rows * cols, filters=filters,
                         patch_size=channels * kh * kw, positions=positions)

            def index():
                return patch_index(channels, rows, cols, kh, kw, stride)

            with self.subTest(shape=shape):
                self.agree(lambda: kernels.conv_forward(image, index(), kernel, **sizes))
                self.agree(lambda: kernels.conv_kernel_grad(image, index(), grad, **sizes))
                self.agree(lambda: kernels.conv_input_grad(grad, kernel, index(), **sizes))

    def test_conv_direct_agrees(self) -> None:
        for batches, channels, rows, cols, filters, kh, kw in [(1, 1, 1, 1, 1, 1, 1), (3, 2, 9, 13, 5, 3, 3),
                                                                (2, 7, 6, 30, 16, 5, 2)]:
            image = values(batches * channels * rows * cols, 4)
            kernel = values(filters * channels * kh * kw, 5)
            with self.subTest(filters=filters):
                self.agree(lambda: kernels.conv_direct(
                    image, kernel, batches=batches, channels=channels, height=rows, width=cols,
                    filters=filters, kernel_rows=kh, kernel_cols=kw))

    def test_padding_and_cropping_agree(self) -> None:
        x = values(3 * 5 * 7, 6)
        sizes = dict(planes=3, rows=5, cols=7, pad_rows=2, pad_cols=1)
        self.agree(lambda: kernels.pad_planes(x, **sizes))
        padded = values(3 * 9 * 9, 7)
        self.agree(lambda: kernels.crop_planes(padded, **sizes))

    def test_indexed_gathers_and_scatters_agree(self) -> None:
        image, grad = values(3 * 50, 8), values(3 * 300, 9)
        positions = [random.Random(10).randrange(50) for _ in range(300)]   # with repeats
        self.agree(lambda: kernels.gather_indexed(image, kernels.index_table(positions), batches=3, image_size=50))
        # Through an index table, the scatter is a gather in the C loop's order: the same bits.
        self.agree(lambda: kernels.scatter_add_indexed(grad, kernels.index_table(positions), batches=3,
                                                       image_size=50), tolerance=0)
        start = values(3 * 50, 11)
        self.agree(lambda: kernels.scatter_add_indexed(grad, kernels.index_table(positions), batches=3,
                                                       image_size=50, into=kernels.storage(start)), tolerance=0)

        def on_the_gpu_only():                    # indices with no host copy: the atomic scatter
            if kernels.BACKEND == "cuda":
                return kernels_cuda._device(array("l", positions), "l")
            return positions

        self.agree(lambda: kernels.scatter_add_indexed(grad, on_the_gpu_only(), batches=3, image_size=50))

    def test_depthwise_agrees(self) -> None:
        for batches, channels, rows, cols, k, stride in [(1, 1, 1, 1, 1, 1), (3, 5, 9, 8, 3, 1), (2, 4, 11, 11, 5, 2)]:
            image = values(batches * channels * rows * cols, 12)
            kernel = values(channels * k * k, 13)
            positions = ((rows - k) // stride + 1) * ((cols - k) // stride + 1)
            grad = values(batches * channels * positions, 14)
            sizes = dict(batches=batches, channels=channels, channel_size=rows * cols, q_size=k * k)
            with self.subTest(channels=channels, k=k):
                self.agree(lambda: kernels.depthwise_forward(image, patch_index(1, rows, cols, k, k, stride),
                                                             kernel, **sizes))
                self.agree(lambda: kernels.depthwise_backward(image, patch_index(1, rows, cols, k, k, stride),
                                                              kernel, grad, **sizes))

    def test_sparse_matmul_agrees(self) -> None:
        rng = random.Random(15)
        rows, inner, n = 37, 23, 19
        entries = sorted((rng.randrange(rows), rng.randrange(inner)) for _ in range(120))
        row_start = [0] * (rows + 1)
        for r, _ in entries:
            row_start[r + 1] += 1
        for r in range(rows):
            row_start[r + 1] += row_start[r]
        cols, weights, dense = [c for _, c in entries], values(len(entries), 16), values(inner * n, 17)
        self.agree(lambda: kernels.csr_matmul(kernels.index_table(row_start), kernels.index_table(cols),
                                              kernels.storage(weights), dense, rows=rows, n=n))

    def test_row_gathers_and_scatters_agree(self) -> None:
        table, ids = values(20 * 7, 18), [3, 0, 19, 3, 3, 7]
        grad, start = values(len(ids) * 7, 19), values(20 * 7, 20)
        self.agree(lambda: kernels.gather_rows(table, kernels.index_table(ids), width=7))
        self.agree(lambda: kernels.scatter_add_rows(grad, kernels.index_table(ids), width=7, rows=20))
        self.agree(lambda: kernels.scatter_add_rows(grad, ids, width=7, rows=20, into=kernels.storage(start)))

    def test_pooling_agrees(self) -> None:
        image = [round(v, 1) for v in values(6 * 9 * 10, 21)]      # rounded: ties inside windows
        for window, stride in [((1, 1), 1), ((2, 2), 2), ((3, 2), 1)]:
            sizes = dict(planes=6, plane_size=90, window_size=window[0] * window[1])
            windows = ((9 - window[0]) // stride + 1) * ((10 - window[1]) // stride + 1)
            grad = values(6 * windows, 22)

            def index():
                return window_index(9, 10, *window, stride)

            with self.subTest(window=window, stride=stride):
                self.agree(lambda: kernels.window_mean(image, index(), **sizes))
                self.agree(lambda: kernels.window_mean_backward(grad, index(), **sizes))
                self.agree(lambda: kernels.window_max(image, index(), **sizes))

    def test_sum_repeated_agrees(self) -> None:
        for outer, size, inner in [(1, 1, 1), (7, 3, 1), (5, 300, 1), (4, 6, 50), (1, 2, 1000)]:
            x = values(outer * size * inner, 23)
            with self.subTest(outer=outer, size=size, inner=inner):
                self.agree(lambda: kernels.sum_repeated(x, outer=outer, size=size, inner=inner))

    def test_max_pool_and_strided_convolution_train_the_same(self) -> None:
        from tensor import conv2d_valid

        def step():
            image = Tensor(values(2 * 3 * 11 * 11, 24), (2, 3, 11, 11), requires_grad=True)
            kernel = Tensor(values(20 * 3 * 3 * 3, 25), (20, 3, 3, 3), requires_grad=True)
            out = conv2d_valid(image, kernel, stride=2).max_pool2d((2, 2)).avg_pool2d((2, 2))
            (out * out).sum().backward()
            return list(out.data), list(image.grad), list(kernel.grad)

        self.agree(step, tolerance=1e-11)

    def test_float32_convolutions(self) -> None:
        batches, channels, rows, cols, filters = 3, 6, 10, 9, 20
        image, kernel = values(batches * channels * rows * cols, 26), values(filters * channels * 9, 27)
        grad = values(batches * filters * 8 * 7, 28)
        sizes = dict(batches=batches, image_size=channels * rows * cols, filters=filters,
                     patch_size=channels * 9, positions=8 * 7)
        cases = {
            "conv_forward": lambda: kernels.conv_forward(image, patch_index(channels, rows, cols, 3, 3), kernel, **sizes),
            "conv_kernel_grad": lambda: kernels.conv_kernel_grad(image, patch_index(channels, rows, cols, 3, 3), grad,
                                                                 **sizes),
            "conv_input_grad": lambda: kernels.conv_input_grad(grad, kernel, patch_index(channels, rows, cols, 3, 3),
                                                               **sizes),
        }
        for name, function in cases.items():
            with self.subTest(kernel=name):
                kernels.use_backend("c")
                expected = flat(function())
                kernels.use_backend("cuda")
                kernels.set_default_dtype("float32")
                try:
                    result = function()
                    self.assertEqual(result.typecode, "f")
                    self.assertClose(expected, flat(result), tolerance=1e-4)
                finally:
                    kernels.set_default_dtype("float64")

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
