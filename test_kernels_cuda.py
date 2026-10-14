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

# (end of rows tests)


# Recurrent: the LSTM cell and layer, the CRF.


def in_float32(function):
    """function run with float32 tensors (on whichever backend is active)."""

    def run():
        kernels.set_default_dtype("float32")
        try:
            return function()
        finally:
            kernels.set_default_dtype("float64")
    return run


class RecurrentTests(CudaTestCase):
    # (batch, hidden) pairs: one value, sizes off the 16 x 16 tiles and the 256-thread blocks
    SHAPES = ((1, 1), (3, 5), (2, 17), (5, 70))

    def test_lstm_cell_kernels_agree(self) -> None:
        for batch, hidden in self.SHAPES:
            gates, state = values(batch * 4 * hidden, 1, -6, 6), values(batch * hidden, 2, -3, 3)
            grad = values(batch * hidden, 3)
            sizes = dict(batch=batch, hidden=hidden)
            with self.subTest(batch=batch, hidden=hidden):
                self.agree(lambda: kernels.lstm_memory(gates, state, **sizes))
                self.agree(lambda: kernels.lstm_output(gates, state, **sizes))
                self.agree(lambda: kernels.lstm_memory_backward(gates, state, grad, **sizes))
                self.agree(lambda: kernels.lstm_output_backward(gates, state, grad, **sizes))

    def lstm_layer(self, batch, steps, hidden):
        """lstm_sequence forward and backward, as plain lists."""

        width = 4 * hidden
        input_gates = kernels.storage(values(batch * steps * width, 4, -2, 2))
        weight = kernels.storage(values(width * hidden, 5, -0.5, 0.5))
        h0, c0 = kernels.storage(values(batch * hidden, 6)), kernels.storage(values(batch * hidden, 7))
        grad = kernels.storage(values(batch * (steps + 1) * hidden, 8))
        sizes = dict(batch=batch, steps=steps, hidden=hidden)
        packed, saved = kernels.lstm_sequence(input_gates, weight, h0, c0, **sizes)
        grads = kernels.lstm_sequence_backward(weight, saved, grad, **sizes)
        return (packed, *saved, *grads)

    def test_lstm_sequence_agrees(self) -> None:
        for batch, steps, hidden in ((1, 1, 1), (3, 4, 5), (2, 7, 17), (5, 3, 70), (17, 2, 9)):
            with self.subTest(batch=batch, steps=steps, hidden=hidden):
                self.agree(lambda: self.lstm_layer(batch, steps, hidden))

    def test_lstm_sequence_in_float32(self) -> None:
        self.agree(in_float32(lambda: self.lstm_layer(3, 6, 21)), tolerance=1e-4)

    def test_lstm_sequence_keeps_its_state_on_the_gpu(self) -> None:
        kernels.use_backend("cuda")
        before = kernels_cuda.syncs
        result = self.lstm_layer(4, 5, 8)
        self.assertEqual(kernels_cuda.syncs, before)
        self.assertTrue(all(isinstance(part, kernels_cuda.DeviceArray) for part in result))

    def crf(self, batch, steps, tags):
        """crf_forward forward and backward, as plain lists."""

        e = kernels.storage(values(batch * steps * tags, 9, -3, 3))
        start = kernels.storage(values(tags, 10))
        weights = kernels.storage(values(tags * tags, 11, 0.1, 3))
        grad = kernels.storage(values(batch * tags, 12))
        sizes = dict(batch=batch, steps=steps, tags=tags)
        alpha, saved = kernels.crf_forward(e, start, weights, **sizes)
        return (alpha, *saved, *kernels.crf_forward_backward(weights, saved, grad, **sizes))

    def test_crf_agrees(self) -> None:
        for batch, steps, tags in ((1, 1, 1), (1, 1, 5), (3, 2, 4), (4, 9, 17), (2, 5, 40), (2, 3, 300)):
            with self.subTest(batch=batch, steps=steps, tags=tags):
                self.agree(lambda: self.crf(batch, steps, tags))

    def test_crf_in_float32(self) -> None:
        self.agree(in_float32(lambda: self.crf(3, 8, 17)), tolerance=1e-4)

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
