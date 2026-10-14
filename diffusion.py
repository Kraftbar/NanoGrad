"""A denoising diffusion model (DDPM) with a small U-Net, sized for MNIST.

Diffusion models (Ho et al., 2020) generate images by learning to undo
noise. Fix a schedule of T small noise levels beta_1 .. beta_T and let
alpha_bar_t = (1 - beta_1) * ... * (1 - beta_t). Then a training image x0
noised to step t is

    x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps,   eps ~ normal(0, 1)

and the network learns to predict eps from (x_t, t): the loss is simply
the mean squared error between the predicted and the true noise. To
generate, start from pure noise x_T and step backwards:

    x_{t-1} = (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_hat) / sqrt(1 - beta_t) + sqrt(beta_t) * z

(no z on the last step). The network is a U-Net: convolutions that halve
the image twice, then double it back, with "skip" connections that hand
each level's features straight across to the decoder. Every block also
gets the time step t, embedded with sines and cosines as in the
transformer, so one network can denoise at every noise level.
"""

from __future__ import annotations

import math
import random

from tensor import Tensor, concat, upsample_nearest2d
from tensor_nn import TensorConv2D, TensorLinear, TensorModule


def silu(x: Tensor) -> Tensor:
    return x * x.sigmoid()


def timestep_embedding(steps: list[int], width: int) -> Tensor:
    """Sinusoidal embedding of integer time steps: (batch, width)."""

    if width % 2:
        raise ValueError("timestep embedding width must be even (sines and cosines)")
    half = width // 2
    values = []
    for step in steps:
        frequencies = [math.exp(-math.log(10000.0) * i / half) for i in range(half)]
        values += [math.sin(step * f) for f in frequencies] + [math.cos(step * f) for f in frequencies]
    return Tensor(values, (len(steps), width))


class ConvBlock(TensorModule):
    """Two 3x3 convolutions with the time embedding added in between, plus a
    residual connection (a 1x1 convolution when the channel count changes)."""

    def __init__(self, in_channels: int, out_channels: int, time_width: int, *, seed: int = 0) -> None:
        self.conv1 = TensorConv2D((out_channels, in_channels, 3, 3), padding=1, seed=seed)
        self.time = TensorLinear(time_width, out_channels, seed=seed + 1)
        self.conv2 = TensorConv2D((out_channels, out_channels, 3, 3), padding=1, seed=seed + 2)
        self.skip = (
            TensorConv2D((out_channels, in_channels, 1, 1), seed=seed + 3) if in_channels != out_channels else None
        )
        self.out_channels = out_channels

    def __call__(self, x: Tensor, time: Tensor) -> Tensor:
        hidden = silu(self.conv1(x))
        hidden = hidden + self.time(time).reshape((x.shape[0], self.out_channels, 1, 1))
        hidden = silu(self.conv2(hidden))
        return hidden + (x if self.skip is None else self.skip(x))

    def parameters(self) -> list[Tensor]:
        layers = [self.conv1, self.time, self.conv2] + ([self.skip] if self.skip is not None else [])
        return [parameter for layer in layers for parameter in layer.parameters()]


class TinyUNet(TensorModule):
    """28x28 -> 14x14 -> 7x7 -> 14x14 -> 28x28, with skip connections."""

    def __init__(self, *, channels: int = 1, width: int = 16, time_width: int = 32, seed: int = 0) -> None:
        wide = 2 * width
        self.time_width = time_width
        self.time_in = TensorLinear(time_width, 2 * time_width, seed=seed)
        self.time_out = TensorLinear(2 * time_width, 2 * time_width, seed=seed + 1)
        t = 2 * time_width
        self.stem = TensorConv2D((width, channels, 3, 3), padding=1, seed=seed + 2)
        self.down1 = ConvBlock(width, width, t, seed=seed + 10)
        self.shrink1 = TensorConv2D((wide, width, 3, 3), stride=2, padding=1, seed=seed + 20)
        self.down2 = ConvBlock(wide, wide, t, seed=seed + 30)
        self.shrink2 = TensorConv2D((wide, wide, 3, 3), stride=2, padding=1, seed=seed + 40)
        self.middle = ConvBlock(wide, wide, t, seed=seed + 50)
        self.up2 = ConvBlock(2 * wide, wide, t, seed=seed + 60)
        self.up1 = ConvBlock(wide + width, width, t, seed=seed + 70)
        self.head = TensorConv2D((channels, width, 3, 3), padding=1, seed=seed + 80)

    def __call__(self, images: Tensor, steps: list[int]) -> Tensor:
        """Predicted noise for noisy images (B, C, 28, 28) at time steps (B,)."""

        time = self.time_out(silu(self.time_in(timestep_embedding(steps, self.time_width))))
        skip1 = self.down1(self.stem(images), time)                      # 28 x 28
        skip2 = self.down2(self.shrink1(skip1), time)                    # 14 x 14
        hidden = self.middle(self.shrink2(skip2), time)                  # 7 x 7
        hidden = self.up2(concat([upsample_nearest2d(hidden), skip2], axis=1), time)
        hidden = self.up1(concat([upsample_nearest2d(hidden), skip1], axis=1), time)
        return self.head(hidden)

    def blocks(self) -> list[TensorModule]:
        return [self.time_in, self.time_out, self.stem, self.down1, self.shrink1, self.down2,
                self.shrink2, self.middle, self.up2, self.up1, self.head]

    def parameters(self) -> list[Tensor]:
        return [parameter for block in self.blocks() for parameter in block.parameters()]


class Diffusion:
    """DDPM's noise schedule, training loss and sampler around a noise-predicting model."""

    def __init__(self, model: TinyUNet, *, steps: int = 200, beta_start: float = 1e-4, beta_end: float = 0.02) -> None:
        if steps < 2:
            raise ValueError("a diffusion schedule needs at least 2 steps")
        self.model = model
        self.steps = steps
        self.betas = [beta_start + (beta_end - beta_start) * i / (steps - 1) for i in range(steps)]
        self.alpha_bars = []
        product = 1.0
        for beta in self.betas:
            product *= 1.0 - beta
            self.alpha_bars.append(product)

    def noisy(self, images: list[float], noise: list[float], steps: list[int], shape) -> Tensor:
        """x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) eps, per image."""

        per_image = len(images) // len(steps)
        values = []
        for index, step in enumerate(steps):
            a, b = math.sqrt(self.alpha_bars[step]), math.sqrt(1.0 - self.alpha_bars[step])
            start = index * per_image
            values += [a * x + b * e for x, e in zip(images[start : start + per_image], noise[start : start + per_image])]
        return Tensor(values, shape)

    def loss(self, images: list[float], shape, rng: random.Random) -> Tensor:
        """Mean squared error of the predicted noise, at random time steps."""

        steps = [rng.randrange(self.steps) for _ in range(shape[0])]
        noise = [rng.gauss(0.0, 1.0) for _ in images]
        predicted = self.model(self.noisy(images, noise, steps, shape), steps)
        error = predicted - Tensor(noise, shape)
        return (error * error).mean()

    def sample(self, shape, rng: random.Random) -> list[float]:
        """Generate images by denoising pure noise, from step T - 1 down to 0."""

        from tensor import no_grad

        x = [rng.gauss(0.0, 1.0) for _ in range(math.prod(shape))]
        with no_grad():
            for step in reversed(range(self.steps)):
                beta, alpha_bar = self.betas[step], self.alpha_bars[step]
                predicted = self.model(Tensor(x, shape), [step] * shape[0]).data
                scale = beta / math.sqrt(1.0 - alpha_bar)
                x = [(value - scale * eps) / math.sqrt(1.0 - beta) for value, eps in zip(x, predicted)]
                if step > 0:
                    x = [value + math.sqrt(beta) * rng.gauss(0.0, 1.0) for value in x]
        return x

    def ddim_sample(self, shape, rng: random.Random, steps: int = 100) -> list[float]:
        """Generate with DDIM (Song et al., 2021, eta = 0): the same trained
        model, but only ``steps`` of the T time steps, evenly spaced, and no
        fresh noise. Each step predicts the clean image from the noise
        estimate, x0 = (x_t - sqrt(1 - alpha_bar_t) eps) / sqrt(alpha_bar_t),
        clipped to the data range [-1, 1] as DDPM's sampler does, then noises
        it back to the earlier step with that same eps:

            x_prev = sqrt(alpha_bar_prev) x0 + sqrt(1 - alpha_bar_prev) eps
        """

        from tensor import no_grad

        times = sorted({round(i * (self.steps - 1) / (steps - 1)) for i in range(steps)}, reverse=True)
        x = [rng.gauss(0.0, 1.0) for _ in range(math.prod(shape))]
        with no_grad():
            for index, step in enumerate(times):
                alpha_bar = self.alpha_bars[step]
                previous = self.alpha_bars[times[index + 1]] if index + 1 < len(times) else 1.0
                predicted = self.model(Tensor(x, shape), [step] * shape[0]).data
                a, b = math.sqrt(alpha_bar), math.sqrt(1.0 - alpha_bar)
                a_prev, b_prev = math.sqrt(previous), math.sqrt(1.0 - previous)
                x = [a_prev * min(1.0, max(-1.0, (value - b * eps) / a)) + b_prev * eps
                     for value, eps in zip(x, predicted)]
        return x
