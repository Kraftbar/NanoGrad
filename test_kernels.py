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

    def test_elementwise_and_row_kernels_agree(self) -> None:
        x = values(24, 8)
        grad = values(24, 9)
        weight = values(6, 10)
        bias = values(6, 11)

        for name, function in (
            ("gelu", lambda: kernels.gelu(x)),
            ("gelu_backward", lambda: kernels.gelu_backward(x, kernels.gelu(x)[1], grad)),
            ("softmax", lambda: kernels.softmax_rows(x, 6)),
            ("softmax_backward", lambda: kernels.softmax_rows_backward(kernels.softmax_rows(x, 6), grad, 6)),
            ("layer_norm", lambda: kernels.layer_norm_rows(x, weight, bias, 6, 1e-5)),
            ("layer_norm_backward", lambda: kernels.layer_norm_rows_backward(
                grad, *kernels.layer_norm_rows(x, weight, bias, 6, 1e-5)[1:], weight, 6, True,
            )),
        ):
            with self.subTest(kernel=name):
                python, c = self.run_both(function)
                if isinstance(python, tuple):
                    for p, q in zip(python, c):
                        self.assertListsClose(p, q)
                else:
                    self.assertListsClose(python, c)

        no_dx = kernels.layer_norm_rows_backward(grad, *kernels.layer_norm_rows(x, weight, bias, 6, 1e-5)[1:], weight, 6, False)
        self.assertIsNone(no_dx[0])

    def test_gemm_large_shapes_use_the_tiled_path_and_agree(self) -> None:
        # Big enough for the register-tiled path (8+ rows, 16+ columns), with
        # leftover rows and columns, a k longer than one block, and batches.
        for m, k, n in ((19, 300, 37), (8, 16, 16), (33, 5, 70)):
            for trans_a in (False, True):
                for trans_b in (False, True):
                    for steps in ((m * k, k * n, m * n), (m * k, k * n, 0)):
                        with self.subTest(m=m, k=k, n=n, trans_a=trans_a, trans_b=trans_b, steps=steps):
                            a, b = values(2 * m * k, m + k), values(2 * k * n, k + n)
                            python, c = self.run_both(lambda: kernels.gemm(
                                a, b, batches=2, m=m, k=k, n=n,
                                a_step=steps[0], b_step=steps[1], c_step=steps[2],
                                trans_a=trans_a, trans_b=trans_b,
                            ))
                            self.assertListsClose(python, c)

    def test_gemm_with_more_rows_than_one_cached_block_agrees(self) -> None:
        m, k, n = 264, 260, 33            # 256-row blocks of A at kb = 256, then a 4-deep k block
        a, b = values(m * k, 5), values(k * n, 6)
        for trans_a in (False, True):
            for trans_b in (False, True):
                with self.subTest(trans_a=trans_a, trans_b=trans_b):
                    python, c = self.run_both(lambda: kernels.gemm(
                        a, b, batches=1, m=m, k=k, n=n, a_step=0, b_step=0, c_step=0,
                        trans_a=trans_a, trans_b=trans_b,
                    ))
                    self.assertListsClose(python, c)

    def test_conv_forward_agrees_with_gather_then_gemm(self) -> None:
        from tensor import _conv2d_patch_index

        for channels, size, kernel_size, filters, stride in ((1, 9, 3, 17, 1), (3, 12, 3, 20, 2), (2, 7, 5, 4, 1)):
            with self.subTest(channels=channels, filters=filters, stride=stride):
                index = _conv2d_patch_index(channels, size, size, kernel_size, kernel_size, stride)
                patch_size = channels * kernel_size * kernel_size
                positions = len(index) // patch_size
                image = values(2 * channels * size * size, 40)
                kernel = values(filters * patch_size, 41)
                sizes = dict(batches=2, image_size=channels * size * size, filters=filters,
                             patch_size=patch_size, positions=positions)
                python, c = self.run_both(lambda: kernels.conv_forward(image, index, kernel, **sizes))
                self.assertListsClose(python, c)
                patches = kernels.gather_indexed(image, index, batches=2, image_size=sizes["image_size"])
                expected = kernels.gemm(kernel, patches, batches=2, m=filters, k=patch_size, n=positions,
                                        a_step=0, b_step=positions * patch_size, c_step=filters * positions,
                                        trans_b=True)
                self.assertListsClose(c, expected)
                grad = values(2 * filters * positions, 42)
                python, c = self.run_both(lambda: kernels.conv_input_grad(grad, kernel, index, **sizes))
                self.assertListsClose(python, c)
                d_patches = kernels.gemm(grad, kernel, batches=2, m=positions, k=filters, n=patch_size,
                                         a_step=filters * positions, b_step=0, c_step=positions * patch_size,
                                         trans_a=True)
                expected_image = kernels.scatter_add_indexed(d_patches, index, batches=2, image_size=sizes["image_size"])
                self.assertListsClose(c, expected_image)
                python, c = self.run_both(lambda: kernels.conv_kernel_grad(image, index, grad, **sizes))
                self.assertListsClose(python, c)
                expected = kernels.gemm(grad, patches, batches=2, m=filters, k=positions, n=patch_size,
                                        a_step=filters * positions, b_step=positions * patch_size, c_step=0)
                self.assertListsClose(c, expected)

    def test_gather_and_scatter_agree(self) -> None:
        image = values(2 * 12, 12)
        index = [0, 5, 5, 11, 3]
        grad = values(2 * 5, 13)

        python, c = self.run_both(lambda: kernels.gather_indexed(image, index, batches=2, image_size=12))
        self.assertListsClose(python, c)
        self.assertEqual(python[:5], [image[i] for i in index])
        python, c = self.run_both(lambda: kernels.scatter_add_indexed(grad, index, batches=2, image_size=12))
        self.assertListsClose(python, c)
        self.assertAlmostEqual(python[5], grad[1] + grad[2])   # index 5 appears twice

    def test_kernel_results_feed_other_kernels(self) -> None:
        image = values(2 * 12, 14)
        index = [0, 5, 5, 11]
        patches = kernels.gather_indexed(image, index, batches=2, image_size=12)
        self.assertIs(type(patches), type(kernels.storage([0.0])))
        product = kernels.gemm(patches, [1.0] * 4, batches=2, m=1, k=4, n=1, a_step=4, b_step=0, c_step=1)
        self.assertAlmostEqual(product[1], sum(image[12 + i] for i in index))

    def test_elementwise_arithmetic_agrees(self) -> None:
        x, y = values(40, 19), values(40, 20)
        for function in (kernels.add, kernels.multiply):
            python, c = self.run_both(lambda: function(x, y))
            self.assertListsClose(python, c)
        python, c = self.run_both(lambda: kernels.scale(x, -2.5))
        self.assertListsClose(python, c)
        self.assertAlmostEqual(python[3], x[3] * -2.5)
        with self.assertRaises(ValueError):
            kernels.add(x, y[:-1])

    def test_adam_update_agrees(self) -> None:
        w, grad = values(30, 21), values(30, 22)
        m, v = values(30, 23), [abs(x) for x in values(30, 24)]
        settings = dict(grad_scale=0.5, beta1=0.9, beta2=0.99, decay=0.999,
                        step_size=1e-3, sqrt_correction=0.3, eps=1e-8)
        python, c = self.run_both(lambda: kernels.adam_update(w, grad, list(m), list(v), **settings))
        for p, q in zip(python, c):
            self.assertListsClose(p, q)

    def test_sum_repeated_agrees(self) -> None:
        x = values(2 * 3 * 4, 27)
        for outer, size, inner in ((2, 3, 4), (6, 4, 1), (1, 24, 1), (24, 1, 1)):
            python, c = self.run_both(lambda: kernels.sum_repeated(x, outer=outer, size=size, inner=inner))
            self.assertListsClose(python, c)
        self.assertAlmostEqual(python[0], sum(x))

    def test_rms_norm_agrees(self) -> None:
        x, weight, grad = values(4 * 6, 28), values(6, 29), values(4 * 6, 30)
        (y_python, r_python), (y_c, r_c) = self.run_both(lambda: kernels.rms_norm_rows(x, weight, 6, 1e-6))
        self.assertListsClose(y_python, y_c)
        self.assertListsClose(r_python, r_c)
        python, c = self.run_both(lambda: kernels.rms_norm_rows_backward(grad, x, r_python, weight, 6))
        for p, q in zip(python, c):
            self.assertListsClose(p, q)

    def test_batch_norm_agrees(self) -> None:
        sizes = dict(batch=3, channels=2, plane=5)
        x, weight, bias, grad = values(30, 31), values(2, 32), values(2, 33), values(30, 34)
        python, c = self.run_both(lambda: kernels.batch_norm_channels(x, weight, bias, eps=1e-5, **sizes))
        for p, q in zip(python, c):
            self.assertListsClose(p, q)
        x_hat, inv_std = python[1], python[2]
        python, c = self.run_both(lambda: kernels.batch_norm_channels_backward(grad, x_hat, inv_std, weight, **sizes))
        for p, q in zip(python, c):
            self.assertListsClose(p, q)

    def test_sigmoid_agrees(self) -> None:
        x = values(40, 35) + [800.0, -800.0, 0.0]          # extremes must not overflow
        grad = values(43, 36)
        python, c = self.run_both(lambda: kernels.sigmoid(x))
        self.assertListsClose(python, c)
        self.assertEqual((python[-3], python[-2], python[-1]), (1.0, 0.0, 0.5))
        python, c = self.run_both(lambda: kernels.sigmoid_backward(python, grad))
        self.assertListsClose(python, c)

    def test_bce_with_logits_sum_agrees(self) -> None:
        x = [value * 40.0 for value in values(30, 37)]       # large |x| must stay stable
        t = [abs(value) for value in values(30, 38)]
        python, c = self.run_both(lambda: kernels.bce_with_logits_sum(x, t))
        self.assertAlmostEqual(python, c, places=9)

    def test_gather_and_scatter_rows_agree(self) -> None:
        table, ids = values(5 * 4, 43), [3, 0, 3, 4]
        python, c = self.run_both(lambda: kernels.gather_rows(table, kernels.index_table(ids), width=4))
        self.assertListsClose(python, c)
        self.assertListsClose(python[:4], table[12:16])
        grad = values(4 * 4, 44)
        python, c = self.run_both(lambda: kernels.scatter_add_rows(grad, kernels.index_table(ids), width=4, rows=5))
        self.assertListsClose(python, c)
        self.assertAlmostEqual(python[12], grad[0] + grad[8])          # id 3 used twice

    def test_add_scaled_and_sum_of_squares_agree(self) -> None:
        x, y = values(50, 41), values(50, 42)
        python, c = self.run_both(lambda: kernels.add_scaled(x, y, -0.3))
        self.assertListsClose(python, c)
        python, c = self.run_both(lambda: kernels.sum_of_squares(x))
        self.assertAlmostEqual(python, c, places=12)

    def test_relu_agrees(self) -> None:
        x, grad = values(40, 25), values(40, 26)
        python, c = self.run_both(lambda: kernels.relu(x))
        self.assertListsClose(python, c)
        python, c = self.run_both(lambda: kernels.relu_backward(x, grad))
        self.assertListsClose(python, c)

    def test_tanh_agrees(self) -> None:
        x = values(50, 15)
        grad = values(50, 16)
        python, c = self.run_both(lambda: kernels.tanh(x))
        self.assertListsClose(python, c)
        python_dx, c_dx = self.run_both(lambda: kernels.tanh_backward(python, grad))
        self.assertListsClose(python_dx, c_dx)

    def test_window_pooling_agrees(self) -> None:
        # Two 3x4 planes; 2x2 windows with stride 1 overlap, so pixels get
        # gradient from several windows. Ties pick the first maximum.
        image = values(2 * 12, 17)
        image[1] = image[0] = 5.0
        index = [0, 1, 4, 5, 1, 2, 5, 6, 6, 7, 10, 11]
        sizes = dict(planes=2, plane_size=12, window_size=4)
        grad = values(2 * 3, 18)

        python, c = self.run_both(lambda: kernels.window_mean(image, index, **sizes))
        self.assertListsClose(python, c)
        self.assertAlmostEqual(python[0], sum(image[i] for i in (0, 1, 4, 5)) / 4)
        python, c = self.run_both(lambda: kernels.window_mean_backward(grad, index, **sizes))
        self.assertListsClose(python, c)
        self.assertAlmostEqual(python[1], (grad[0] + grad[1]) / 4)

        (python_max, python_at), (c_max, c_at) = self.run_both(lambda: kernels.window_max(image, index, **sizes))
        self.assertListsClose(python_max, c_max)
        self.assertEqual(list(python_at), list(c_at))
        self.assertEqual(python_at[0], 0)
        self.assertEqual(python_at[3], 12 + max((0, 1, 4, 5), key=lambda i: image[12 + i]))

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
