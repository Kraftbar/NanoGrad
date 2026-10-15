"""Mixed precision (kernels.set_matmul_precision) and loss scaling (optim.GradScaler).

On the CPU the switch emulates the tensor cores by rounding the products'
inputs to 16 bits; test_kernels_cuda.py checks the GPU against that.
"""

import math
import random
import unittest

import kernels
import kernels_reference
from optim import GradScaler, TensorSGD
from tensor import Tensor
from tensor_nn import TensorLinear


def values(count, seed, low=-1.0, high=1.0):
    rng = random.Random(seed)
    return [rng.uniform(low, high) for _ in range(count)]


EDGES = [0.0, 1.0, 1 + 2 ** -8, 1 + 3 * 2 ** -8, 1 + 2 ** -11, 65504.0, 65519.9, 65520.0, -1e6, 3e-8, -2.9e-8,
         1e-8, 6.1e-5, 1e-40, 3.4e38, 1e39, float("inf"), float("-inf")]


class RoundingTests(unittest.TestCase):
    def test_known_values(self) -> None:
        bf16 = kernels_reference.round_to_16bit([1 + 2 ** -8, 1 + 3 * 2 ** -8, 70000.0, -1e-30, 1e39], "bf16")
        self.assertEqual(bf16[:3], [1.0, 1 + 2 ** -6, 70144.0])                # ties go to even
        self.assertAlmostEqual(bf16[3], -1e-30, delta=1e-32)                    # float32's range
        self.assertEqual(bf16[4], math.inf)                                     # past float32
        fp16 = kernels_reference.round_to_16bit([1 + 2 ** -11, 65519.9, 65520.0, 3e-8, 2.9e-8], "fp16")
        self.assertEqual(fp16, [1.0, 65504.0, math.inf, 2 ** -24, 0.0])        # 2^-24: the smallest step
        self.assertTrue(math.isnan(kernels_reference.round_to_16bit([math.nan], "bf16")[0]))

    @unittest.skipUnless(kernels.BACKEND in ("c", "cuda"), "needs the C kernels")
    def test_c_agrees_with_the_reference(self) -> None:
        numbers = EDGES + [v * 10 ** e for e, v in zip(range(-12, 12), values(24, 5, -3.0, 3.0))] + values(300, 6)
        previous = kernels.BACKEND
        kernels.use_backend("c")
        try:
            for kind in ("bf16", "fp16"):
                self.assertEqual(list(kernels.round_to_16bit(numbers, kind)),
                                 kernels_reference.round_to_16bit(numbers, kind))
        finally:
            kernels.use_backend(previous)


class EmulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = kernels.BACKEND
        if kernels.BACKEND == "cuda":                  # (the emulation is the CPU backends')
            kernels.use_backend("c")

    def tearDown(self) -> None:
        kernels.set_matmul_precision("fp32")
        kernels.use_backend(self.previous)

    def test_products_read_rounded_inputs(self) -> None:
        a, b = values(7 * 40, 1), values(40 * 9, 2)
        sizes = dict(batches=1, m=7, k=40, n=9, a_step=0, b_step=0, c_step=0, trans_b=True)
        for kind in ("bf16", "fp16"):
            expected = list(kernels.gemm(kernels.round_to_16bit(a, kind), kernels.round_to_16bit(b, kind), **sizes))
            with kernels.autocast(kind):
                self.assertEqual(list(kernels.gemm(a, b, **sizes)), expected)
            self.assertNotEqual(list(kernels.gemm(a, b, **sizes)), expected)

    def test_convolutions_read_rounded_inputs(self) -> None:
        from tensor import conv2d

        image, kernel = values(2 * 3 * 6 * 6, 3), values(20 * 3 * 3 * 3, 4)     # 20 filters: the matrix product
        def run(kind=None):
            x = Tensor(image if kind is None else kernels.round_to_16bit(image, kind), (2, 3, 6, 6))
            w = Tensor(kernel if kind is None else kernels.round_to_16bit(kernel, kind), (20, 3, 3, 3))
            return list(conv2d(x, w, padding=1).data)
        with kernels.autocast("bf16"):
            mixed = run()
        self.assertEqual(mixed, run("bf16"))
        self.assertLess(max(abs(p - q) for p, q in zip(mixed, run())), 0.05)

    def test_the_python_backend_emulates_too(self) -> None:
        kernels.use_backend("python")
        a, b = values(3 * 5, 1), values(5 * 2, 2)
        sizes = dict(batches=1, m=3, k=5, n=2, a_step=0, b_step=0, c_step=0)
        expected = kernels.gemm(kernels.round_to_16bit(a, "bf16"), kernels.round_to_16bit(b, "bf16"), **sizes)
        with kernels.autocast("bf16"):
            self.assertEqual(kernels.gemm(a, b, **sizes), expected)

    def test_the_switch(self) -> None:
        self.assertEqual(kernels.matmul_precision(), "fp32")
        with kernels.autocast():
            self.assertEqual(kernels.matmul_precision(), "bf16")
        self.assertEqual(kernels.matmul_precision(), "fp32")
        with self.assertRaises(ValueError):
            kernels.set_matmul_precision("float16")


class GradScalerTests(unittest.TestCase):
    """Gradients of 1e-9 vanish in float16 (its smallest step is 6e-8); scaled, they survive."""

    def tearDown(self) -> None:
        kernels.set_matmul_precision("fp32")

    def tiny_gradients(self, scaler=None):
        layer = TensorLinear(30, 20, seed=0)
        x = Tensor(values(8 * 30, 1), (8, 30))
        loss = layer(x).sum() * 1e-9
        (loss if scaler is None else scaler.scale(loss)).backward()
        return layer

    def test_tiny_gradients_vanish_without_scaling_and_survive_with_it(self) -> None:
        exact = list(self.tiny_gradients().weight.grad)
        kernels.set_matmul_precision("fp16")
        self.assertEqual(set(self.tiny_gradients().weight.grad), {0.0})
        scaler = GradScaler()
        layer = self.tiny_gradients(scaler)
        before = list(layer.weight.data)
        self.assertTrue(scaler.step(TensorSGD(layer.parameters(), lr=1e8)))
        scaler.update()
        moved = [(b - a) / 1e8 for a, b in zip(layer.weight.data, before)]      # lr * unscaled gradient
        for got, want in zip(moved, exact):
            self.assertAlmostEqual(got, want, delta=1e-3 * max(map(abs, exact)))

    def test_an_overflow_skips_the_step_and_halves_the_scale(self) -> None:
        kernels.set_matmul_precision("fp16")
        layer = TensorLinear(4, 3, seed=0)
        scaler = GradScaler(init_scale=2.0 ** 30)
        loss = layer(Tensor(values(2 * 4, 1), (2, 4))).sum()
        scaler.scale(loss).backward()                          # 2^30 is past float16's 65504: infinite
        before = list(layer.weight.data)
        self.assertFalse(scaler.step(TensorSGD(layer.parameters(), lr=0.1)))
        scaler.update()
        self.assertEqual(list(layer.weight.data), before)
        self.assertEqual(scaler.scale_factor, 2.0 ** 29)
        self.assertEqual(scaler.skipped_steps, 1)

    def test_the_scale_grows_after_good_steps(self) -> None:
        layer = TensorLinear(4, 3, seed=0)
        optimizer = TensorSGD(layer.parameters(), lr=0.1)
        scaler = GradScaler(init_scale=8.0, growth_interval=3)
        for _ in range(3):
            optimizer.zero_grad()
            scaler.scale(layer(Tensor(values(2 * 4, 1), (2, 4))).sum()).backward()
            scaler.step(optimizer)
            scaler.update()
        self.assertEqual(scaler.scale_factor, 16.0)

    def test_disabled_it_is_plain_training(self) -> None:
        def train(scaler):
            layer = TensorLinear(4, 3, seed=0)
            optimizer = TensorSGD(layer.parameters(), lr=0.1)
            loss = layer(Tensor(values(2 * 4, 1), (2, 4))).sum()
            (loss if scaler is None else scaler.scale(loss)).backward()
            optimizer.step() if scaler is None else scaler.step(optimizer)
            return list(layer.weight.data)
        self.assertEqual(train(GradScaler(enabled=False)), train(None))


if __name__ == "__main__":
    unittest.main()
