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
from language import TransformerBlock
from tensor_nn import TensorConv2D, TensorGroupNorm, TensorLinear, TensorModule


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
    residual connection (a 1x1 convolution when the channel count changes).

    With ``groups`` it is DDPM's residual block: group norm and SiLU before
    each convolution (GN -> SiLU -> conv, + time, GN -> SiLU -> conv)."""

    def __init__(self, in_channels: int, out_channels: int, time_width: int, *, groups: int = 0, seed: int = 0) -> None:
        self.conv1 = TensorConv2D((out_channels, in_channels, 3, 3), padding=1, seed=seed)
        self.time = TensorLinear(time_width, out_channels, seed=seed + 1)
        self.conv2 = TensorConv2D((out_channels, out_channels, 3, 3), padding=1, seed=seed + 2)
        self.skip = (
            TensorConv2D((out_channels, in_channels, 1, 1), seed=seed + 3) if in_channels != out_channels else None
        )
        self.norms = [TensorGroupNorm(groups, in_channels), TensorGroupNorm(groups, out_channels)] if groups else []
        self.out_channels = out_channels

    def __call__(self, x: Tensor, time: Tensor) -> Tensor:
        time = self.time(time).reshape((x.shape[0], self.out_channels, 1, 1))
        if self.norms:
            hidden = self.conv1(silu(self.norms[0](x))) + time
            hidden = self.conv2(silu(self.norms[1](hidden)))
        else:
            hidden = silu(self.conv1(x)) + time
            hidden = silu(self.conv2(hidden))
        return hidden + (x if self.skip is None else self.skip(x))

    def parameters(self) -> list[Tensor]:
        layers = [self.conv1, self.time, self.conv2] + ([self.skip] if self.skip is not None else []) + self.norms
        return [parameter for layer in layers for parameter in layer.parameters()]


class TinyUNet(TensorModule):
    """28x28 -> 14x14 -> 7x7 -> 14x14 -> 28x28, with skip connections."""

    def __init__(self, *, channels: int = 1, width: int = 16, time_width: int = 32, groups: int = 0,
                 attention: bool = False, seed: int = 0) -> None:
        wide = 2 * width
        self.time_width = time_width
        self.time_in = TensorLinear(time_width, 2 * time_width, seed=seed)
        self.time_out = TensorLinear(2 * time_width, 2 * time_width, seed=seed + 1)
        t = 2 * time_width
        self.stem = TensorConv2D((width, channels, 3, 3), padding=1, seed=seed + 2)
        self.down1 = ConvBlock(width, width, t, groups=groups, seed=seed + 10)
        self.shrink1 = TensorConv2D((wide, width, 3, 3), stride=2, padding=1, seed=seed + 20)
        self.down2 = ConvBlock(wide, wide, t, groups=groups, seed=seed + 30)
        self.shrink2 = TensorConv2D((wide, wide, 3, 3), stride=2, padding=1, seed=seed + 40)
        self.middle = ConvBlock(wide, wide, t, groups=groups, seed=seed + 50)
        self.up2 = ConvBlock(2 * wide, wide, t, groups=groups, seed=seed + 60)
        self.up1 = ConvBlock(wide + width, width, t, groups=groups, seed=seed + 70)
        self.head = TensorConv2D((channels, width, 3, 3), padding=1, seed=seed + 80)
        # Self-attention over the 7 x 7 positions of the middle (DDPM attends
        # at its 16 x 16 level): a non-causal pre-norm transformer block, so
        # every position can see the whole digit, not just its neighbours.
        self.attend = (TransformerBlock(wide, 49, num_heads=1, feed_forward_activation="gelu", causal=False,
                                        seed=seed + 90) if attention else None)

    def __call__(self, images: Tensor, steps: list[int]) -> Tensor:
        """Predicted noise for noisy images (B, C, 28, 28) at time steps (B,)."""

        time = self.time_out(silu(self.time_in(timestep_embedding(steps, self.time_width))))
        skip1 = self.down1(self.stem(images), time)                      # 28 x 28
        skip2 = self.down2(self.shrink1(skip1), time)                    # 14 x 14
        hidden = self.middle(self.shrink2(skip2), time)                  # 7 x 7
        if self.attend is not None:
            batch, wide = hidden.shape[:2]
            tokens = self.attend(hidden.reshape((batch, wide, 49)).permute((0, 2, 1)))
            hidden = tokens.permute((0, 2, 1)).reshape((batch, wide, 7, 7))
        hidden = self.up2(concat([upsample_nearest2d(hidden), skip2], axis=1), time)
        hidden = self.up1(concat([upsample_nearest2d(hidden), skip1], axis=1), time)
        return self.head(hidden)

    def blocks(self) -> list[TensorModule]:
        return [self.time_in, self.time_out, self.stem, self.down1, self.shrink1, self.down2,
                self.shrink2, self.middle, self.up2, self.up1, self.head] + ([self.attend] if self.attend else [])

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

    def sample(self, shape, rng: random.Random, *, clip: bool = True) -> list[float]:
        """Generate images by denoising pure noise, from step T - 1 down to 0.

        With ``clip`` (as in Ho et al.'s released code) each step first
        predicts the clean image x0 = (x_t - sqrt(1 - alpha_bar_t) eps) /
        sqrt(alpha_bar_t), clips it to [-1, 1], and steps to the mean of
        q(x_{t-1} | x_t, x0),

            sqrt(alpha_bar_{t-1}) beta_t / (1 - alpha_bar_t) x0
              + sqrt(1 - beta_t) (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) x_t,

        which without the clip is the same as the formula in the module
        docstring. Fresh noise of variance beta_t is added on every step but
        the last.
        """

        from tensor import no_grad

        x = [rng.gauss(0.0, 1.0) for _ in range(math.prod(shape))]
        with no_grad():
            for step in reversed(range(self.steps)):
                beta, alpha_bar = self.betas[step], self.alpha_bars[step]
                previous = self.alpha_bars[step - 1] if step > 0 else 1.0
                predicted = self.model(Tensor(x, shape), [step] * shape[0]).data
                if clip:
                    a, b = math.sqrt(alpha_bar), math.sqrt(1.0 - alpha_bar)
                    clean_weight = math.sqrt(previous) * beta / (1.0 - alpha_bar)
                    x_weight = math.sqrt(1.0 - beta) * (1.0 - previous) / (1.0 - alpha_bar)
                    x = [clean_weight * min(1.0, max(-1.0, (value - b * eps) / a)) + x_weight * value
                         for value, eps in zip(x, predicted)]
                else:
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
        it back to the earlier step:

            x_prev = sqrt(alpha_bar_prev) x0 + sqrt(1 - alpha_bar_prev) eps'

        where eps' = (x_t - sqrt(alpha_bar_t) x0) / sqrt(1 - alpha_bar_t) is the
        noise that the clipped x0 implies. Re-noising with the model's own eps
        instead lets x drift away from x0 whenever the clip is active, and
        nothing pulls it back: x grows geometrically until it overflows.
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
                clean = [min(1.0, max(-1.0, (value - b * eps) / a)) for value, eps in zip(x, predicted)]
                x = [a_prev * x0 + b_prev * (value - a * x0) / b for value, x0 in zip(x, clean)]
        return x
