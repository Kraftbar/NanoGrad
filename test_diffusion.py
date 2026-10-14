import importlib.util
import random
import unittest

from diffusion import Diffusion, TinyUNet, timestep_embedding
from gradcheck import max_gradient_error
from tensor import Tensor, upsample_nearest2d


class DiffusionTests(unittest.TestCase):
    def test_upsample_values_and_gradient(self) -> None:
        image = Tensor([1.0, 2.0, 3.0, 4.0], (1, 1, 2, 2), requires_grad=True)
        self.assertEqual(upsample_nearest2d(image).tolist()[0][0][1], [1.0, 1.0, 2.0, 2.0])
        weights = Tensor([float(i) for i in range(16)], (1, 1, 4, 4))
        self.assertLess(max_gradient_error(lambda: (upsample_nearest2d(image) * weights).sum(), [image]), 1e-6)

    def test_schedule_and_noising(self) -> None:
        process = Diffusion(TinyUNet(width=4, time_width=8), steps=10)
        self.assertAlmostEqual(process.alpha_bars[0], 1 - 1e-4)
        self.assertLess(process.alpha_bars[-1], process.alpha_bars[0])
        noisy = process.noisy([1.0, 1.0], [0.0, 0.0], [0], (1, 1, 1, 2))
        self.assertAlmostEqual(noisy.data[0], process.alpha_bars[0] ** 0.5)

    def test_unet_shapes_and_sampling(self) -> None:
        model = TinyUNet(width=4, time_width=8)
        rng = random.Random(1)
        out = model(Tensor([rng.random() for _ in range(2 * 784)], (2, 1, 28, 28)), [3, 7])
        self.assertEqual(out.shape, (2, 1, 28, 28))
        samples = Diffusion(model, steps=3).sample((1, 1, 28, 28), rng)
        self.assertEqual(len(samples), 784)
        self.assertEqual(timestep_embedding([0, 5], 8).shape, (2, 8))

    def test_png_writer(self) -> None:
        import struct
        import tempfile
        import zlib
        from pathlib import Path

        from features import save_png

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            save_png(path, Tensor([0.0, 0.5, 1.0, 0.25, 0.75, 1.0], (2, 3)))
            data = path.read_bytes()
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", data[16:24])
        self.assertEqual((width, height), (3, 2))
        start = data.index(b"IDAT") + 4
        length = struct.unpack(">I", data[start - 8 : start - 4])[0]
        self.assertEqual(zlib.decompress(data[start : start + length]), bytes([0, 0, 128, 255, 0, 64, 191, 255]))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_unet_matches_pytorch(self) -> None:
        import torch

        from torch_twin import TorchTwin

        model = TinyUNet(width=4, time_width=8, seed=2)
        twin = TorchTwin(model)
        rng = random.Random(3)
        pixels = [rng.uniform(-1.0, 1.0) for _ in range(2 * 784)]
        steps = [5, 150]
        weights = [rng.uniform(-1.0, 1.0) for _ in range(2 * 784)]
        ours = model(Tensor(pixels, (2, 1, 28, 28)), steps)
        (ours * Tensor(weights, ours.shape)).sum().backward()
        theirs = twin.unet(model, torch.tensor(pixels, dtype=torch.float64).reshape(2, 1, 28, 28), steps)
        (theirs * torch.tensor(weights, dtype=torch.float64).reshape(theirs.shape)).sum().backward()
        for a, b in zip(ours.data, theirs.detach().reshape(-1).tolist()):
            self.assertAlmostEqual(a, b, places=10)
        for parameter, tensor in zip(model.parameters(), twin.parameters):
            for a, b in zip(parameter.grad, tensor.grad.reshape(-1).tolist()):
                self.assertAlmostEqual(a, b, places=9)


class SamplerTests(unittest.TestCase):
    def test_ddim_with_a_perfect_noise_guess_returns_the_image(self) -> None:
        # A model that knows the clean image x0 can predict the noise exactly:
        # eps = (x_t - sqrt(alpha_bar) x0) / sqrt(1 - alpha_bar). DDIM then
        # recovers x0 at every step, so the sample is x0 itself.
        target = [0.5, -0.25, 0.75, 0.0]

        class Oracle:
            def __call__(self, images: Tensor, steps: list[int]) -> Tensor:
                a = process.alpha_bars[steps[0]]
                return Tensor([(x - a ** 0.5 * x0) / (1 - a) ** 0.5 for x, x0 in zip(images.data, target)],
                              images.shape)

        process = Diffusion(TinyUNet(width=4, time_width=8), steps=50)
        process.model = Oracle()
        samples = process.ddim_sample((1, 1, 2, 2), random.Random(0), steps=10)
        for value, expected in zip(samples, target):
            self.assertAlmostEqual(value, expected, places=9)

    def test_ddpm_posterior_mean_matches_the_noise_form_without_clipping(self) -> None:
        # Inside the data range the clip does nothing, so the x0 / posterior
        # mean form must give the same samples as the eps form.
        # The oracle's noise estimate always implies the same in-range x0.
        target = [0.5, -0.25, 0.75, 0.0]

        class Oracle:
            def __call__(self, images: Tensor, steps: list[int]) -> Tensor:
                a = process.alpha_bars[steps[0]]
                return Tensor([(x - a ** 0.5 * x0) / (1 - a) ** 0.5 for x, x0 in zip(images.data, target)],
                              images.shape)

        process = Diffusion(TinyUNet(width=4, time_width=8), steps=5, beta_start=0.1, beta_end=0.2)
        process.model = Oracle()
        clipped = process.sample((1, 1, 2, 2), random.Random(3))
        plain = process.sample((1, 1, 2, 2), random.Random(3), clip=False)
        for a, b in zip(clipped, plain):
            self.assertAlmostEqual(a, b, places=12)

    def test_ddim_stays_bounded_when_the_clip_is_active(self) -> None:
        # A model whose noise estimate does not match x (here eps = 1e6 x)
        # keeps the x0 clip active. Re-noising with that eps made x grow
        # geometrically until it overflowed and the sample came out NaN;
        # with the eps that the clipped x0 implies, x stays in the data range.
        class Inconsistent:
            def __call__(self, images: Tensor, steps: list[int]) -> Tensor:
                return Tensor([1e6 * x for x in images.data], images.shape)

        process = Diffusion(TinyUNet(width=4, time_width=8), steps=1000)
        process.model = Inconsistent()
        samples = process.ddim_sample((1, 1, 2, 2), random.Random(0), steps=100)
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in samples))

    def test_ema_tracks_and_swaps(self) -> None:
        from diffusion_demo import EMA

        weight = Tensor([0.0, 2.0], (2,), requires_grad=True)
        ema = EMA([weight], decay=0.5)
        weight.data = type(weight.data)(weight.data.typecode, [4.0, 2.0]) if hasattr(weight.data, "typecode") \
            else [4.0, 2.0]
        ema.update()
        self.assertEqual(list(ema.shadow[0]), [2.0, 2.0])
        ema.swap()
        self.assertEqual(list(weight.data), [2.0, 2.0])
        ema.swap()
        self.assertEqual(list(weight.data), [4.0, 2.0])


class FrechetDistanceTests(unittest.TestCase):
    def test_known_gaussians(self) -> None:
        from sample_quality import frechet_distance

        # Equal covariances: only the mean term remains.
        same = [[2.0, 0.5], [0.5, 1.0]]
        self.assertAlmostEqual(frechet_distance(([0.0, 0.0], same), ([3.0, 4.0], same)), 25.0, places=9)
        # Diagonal covariances: sum over axes of (sqrt(a) - sqrt(b))^2.
        first, second = [[4.0, 0.0], [0.0, 9.0]], [[1.0, 0.0], [0.0, 1.0]]
        self.assertAlmostEqual(frechet_distance(([0.0, 0.0], first), ([0.0, 0.0], second)), 1.0 + 4.0, places=9)


if __name__ == "__main__":
    unittest.main()
