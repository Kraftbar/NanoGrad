import random
import unittest

import kernels
from tensor import Tensor


def values(count, seed):
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(count)]


@unittest.skipUnless(kernels.BACKEND == "c", "C kernels are not built here")
class Float32Tests(unittest.TestCase):
    def tearDown(self) -> None:
        kernels.set_default_dtype("float64")

    def test_tensors_store_float32_arrays(self) -> None:
        kernels.set_default_dtype("float32")
        self.assertEqual(kernels.default_dtype(), "float32")
        tensor = Tensor([1.0, 2.5, 3.0], (3,), requires_grad=True)
        self.assertEqual(tensor.data.typecode, "f")
        self.assertEqual((tensor * 2.0).data.typecode, "f")
        kernels.set_default_dtype("float64")
        self.assertEqual(Tensor([1.0], (1,)).data.typecode, "d")

    def test_float32_kernels_agree_with_the_reference(self) -> None:
        a, b = values(19 * 40, 1), values(40 * 37, 2)
        expected = kernels.reference.gemm(a, b, 1, 19, 40, 37, 0, 0, 0, False, False)
        kernels.set_default_dtype("float32")
        got = kernels.gemm(a, b, batches=1, m=19, k=40, n=37, a_step=0, b_step=0, c_step=0)
        self.assertEqual(got.typecode, "f")
        for x, y in zip(got, expected):
            self.assertAlmostEqual(x, y, delta=1e-5 * (1 + abs(y)))
        x = values(4 * 9, 3)
        expected = kernels.reference.softmax_rows(x, 9)
        for p, q in zip(kernels.softmax_rows(x, 9), expected):
            self.assertAlmostEqual(p, q, delta=1e-6)

    def test_a_training_step_tracks_float64(self) -> None:
        from mnist_compare import NanoGradRunner, build_nanograd

        rng = random.Random(4)
        pixels = [rng.random() for _ in range(8 * 784)]
        labels = [rng.randrange(10) for _ in range(8)]
        losses = {}
        for dtype in ("float64", "float32"):
            kernels.set_default_dtype(dtype)
            runner = NanoGradRunner(build_nanograd(0, "resnet"), lr=1e-3)
            losses[dtype] = [runner.train_step(pixels, labels) for _ in range(3)]
        for x, y in zip(losses["float32"], losses["float64"]):
            self.assertAlmostEqual(x, y, delta=1e-4)
        self.assertLess(losses["float32"][-1], losses["float32"][0])


    def test_cached_rotary_tables_follow_the_current_precision(self) -> None:
        import math

        from tensor import rotary_embedding

        x = Tensor([1.0, 0.0] * 6, (1, 6, 2))
        kernels.set_default_dtype("float32")
        rotary_embedding(x)                          # fills the cache under float32
        kernels.set_default_dtype("float64")
        out = rotary_embedding(Tensor([1.0, 0.0] * 6, (1, 6, 2)))
        for position in range(6):
            self.assertEqual(out.data[2 * position], math.cos(position))   # exact in float64

    def test_gpt_train_keeps_the_environment_dtype_unless_asked(self) -> None:
        from gpt_train import parse_args

        self.assertIsNone(parse_args(["--text-file", "t.txt"]).dtype)
        self.assertEqual(parse_args(["--text-file", "t.txt", "--dtype", "float32"]).dtype, "float32")


if __name__ == "__main__":
    unittest.main()
