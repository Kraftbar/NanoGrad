"""Long-horizon time-series forecasting on ETTh1 with DLinear, NanoGrad vs PyTorch.

ETTh1 (Zhou et al., Informer, 2021) is two years of hourly readings from
an electricity transformer: oil temperature and six load measurements.
From the last 336 hours of all 7 series, predict the next 96 (or 192,
336, 720). The standard protocol: train on the first 12 months, validate
on the next 4, test on the 4 after; standardize each series with the
training months' mean and spread; report mean squared error on the
standardized values.

DLinear (Zeng et al., "Are Transformers Effective for Time Series
Forecasting?", 2023) beat the forecasting transformers of the time with
almost nothing:

    trend    = moving average of the input (window 25, ends padded by repeating)
    seasonal = input - trend
    forecast = Linear_s(seasonal) + Linear_t(trend)     each a 336 -> 96 map,
                                                        shared by all 7 series

The moving average is itself a fixed linear map (a 336 x 336 matrix of
1/25s), so the whole model is three matrix products. The paper reports
test MSE 0.375 / 0.405 / 0.439 / 0.472 at horizons 96 / 192 / 336 / 720.
Training follows its code: Adam at lr 0.005 halved every epoch, batch 32,
up to 10 epochs, stopping when validation loss has not improved for 3.

    python3 timeseries.py --horizons 96 192 336 720
    .venv/bin/python timeseries.py --frameworks nanograd torch --horizons 96
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from time import perf_counter

from optim import TensorAdam
from tensor import Tensor, matmul, no_grad
from tensor_nn import TensorLinear, TensorModule


MONTH = 30 * 24                    # hours, as the benchmark counts them


def load_ett(path: str | Path) -> list[list[float]]:
    """Rows of the 7 series (the date column dropped)."""

    with Path(path).open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))[1:]
    return [[float(value) for value in row[1:]] for row in rows]


def standardize(rows: list[list[float]], train_rows: int) -> list[list[float]]:
    columns = len(rows[0])
    means = [sum(row[c] for row in rows[:train_rows]) / train_rows for c in range(columns)]
    spreads = [math.sqrt(sum((row[c] - means[c]) ** 2 for row in rows[:train_rows]) / train_rows) for c in range(columns)]
    return [[(row[c] - means[c]) / spreads[c] for c in range(columns)] for row in rows]


def splits(rows: list[list[float]], lookback: int, horizon: int) -> dict[str, list[tuple[int, int]]]:
    """(start, end) row ranges of each split, each window reaching back ``lookback`` rows."""

    starts = [0, 12 * MONTH - lookback, 16 * MONTH - lookback]
    ends = [12 * MONTH, 16 * MONTH, 20 * MONTH]
    return {name: (start, end) for name, start, end in zip(("train", "validation", "test"), starts, ends)}


def windows(rows: list[list[float]], start: int, end: int, lookback: int, horizon: int) -> list[int]:
    """The first row of every (lookback + horizon)-row window inside rows[start:end]."""

    return list(range(start, end - lookback - horizon + 1))


def batch(rows, firsts: list[int], lookback: int, horizon: int) -> tuple[list[float], list[float]]:
    """Inputs (B, channels, lookback) and targets (B, channels, horizon), series by series."""

    channels = len(rows[0])
    inputs = [rows[t][c] for first in firsts for c in range(channels) for t in range(first, first + lookback)]
    targets = [rows[t][c] for first in firsts for c in range(channels)
               for t in range(first + lookback, first + lookback + horizon)]
    return inputs, targets


def moving_average_matrix(length: int, window: int) -> list[float]:
    """(length x length) A with trend = x @ A^T: each output is the mean of the
    ``window`` inputs centred on it, the ends padded by repeating the first
    and last value (DLinear's moving_avg)."""

    half = (window - 1) // 2
    a = [0.0] * (length * length)
    for t in range(length):
        for offset in range(-half, half + 1):
            a[t * length + min(length - 1, max(0, t + offset))] += 1.0 / window
    return a


class DLinear(TensorModule):
    def __init__(self, lookback: int, horizon: int, *, window: int = 25, seed: int = 0) -> None:
        self.lookback, self.horizon = lookback, horizon
        a = moving_average_matrix(lookback, window)
        self.average_t = Tensor([a[j * lookback + i] for i in range(lookback) for j in range(lookback)],
                                (lookback, lookback))                           # A^T, so trend = x @ A^T
        self.seasonal = TensorLinear(lookback, horizon, seed=seed)
        self.trend = TensorLinear(lookback, horizon, seed=seed + 1)
        bound = 1 / math.sqrt(lookback)                                         # nn.Linear's default init
        rng = random.Random(seed)
        for layer in (self.seasonal, self.trend):
            layer.weight.data = [rng.uniform(-bound, bound) for _ in range(layer.weight.numel)]
            layer.bias.data = [rng.uniform(-bound, bound) for _ in range(layer.bias.numel)]

    def __call__(self, series: Tensor) -> Tensor:
        """(B, channels, lookback) -> (B, channels, horizon)."""

        trend = matmul(series, self.average_t)
        return self.seasonal(series - trend) + self.trend(trend)

    def parameters(self) -> list[Tensor]:
        return [*self.seasonal.parameters(), *self.trend.parameters()]


class PatchTST(TensorModule):
    """A transformer forecaster in the style of PatchTST (Nie et al., 2023).

    Each series is handled on its own ("channel independence"): it is
    normalized by its own window's mean and spread (RevIN), cut into
    overlapping patches of 16 steps (stride 8; 42 patches of 336 steps),
    and each patch becomes one token: linear embedding plus a learned
    position, then the bidirectional transformer blocks of the language
    models (language.py), then one linear map from all patch outputs to the
    forecast, which is de-normalized.
    """

    def __init__(self, lookback: int, horizon: int, *, patch: int = 16, stride: int = 8, width: int = 16,
                 heads: int = 4, layers: int = 3, hidden: int = 128, dropout: float = 0.2, seed: int = 0) -> None:
        from language import TransformerBlock

        self.lookback, self.horizon, self.patch, self.stride = lookback, horizon, patch, stride
        self.patches = (lookback - patch) // stride + 1
        self.embed = TensorLinear(patch, width, seed=seed)
        rng = random.Random(seed + 1)
        self.position = Tensor([rng.uniform(-0.02, 0.02) for _ in range(self.patches * width)],
                               (self.patches, width), requires_grad=True)
        self.blocks = [TransformerBlock(width, self.patches, hidden_dim=hidden, num_heads=heads, causal=False,
                                        feed_forward_activation="gelu", dropout=dropout, seed=seed + 10 * (i + 1))
                       for i in range(layers)]
        self.head = TensorLinear(self.patches * width, horizon, seed=seed + 100)
        self.width = width

    def __call__(self, series: Tensor) -> Tensor:
        """(B, channels, lookback) -> (B, channels, horizon)."""

        batch, channels, length = series.shape
        rows, values = batch * channels, series.data
        # RevIN: each window by its own mean and spread (constants: no gradient).
        means, spreads, patches = [], [], []
        for row in range(rows):
            window = values[row * length : (row + 1) * length]
            mean = sum(window) / length
            spread = math.sqrt(sum((v - mean) ** 2 for v in window) / length + 1e-5)
            means.append(mean)
            spreads.append(spread)
            for p in range(self.patches):
                start = p * self.stride
                patches += [(v - mean) / spread for v in window[start : start + self.patch]]
        tokens = self.embed(Tensor(patches, (rows, self.patches, self.patch))) + self.position
        for block in self.blocks:
            tokens = block(tokens)
        forecast = self.head(tokens.reshape((rows, self.patches * self.width)))               # (rows, horizon)
        forecast = forecast * Tensor(spreads, (rows, 1)) + Tensor(means, (rows, 1))
        return forecast.reshape((batch, channels, self.horizon))

    def parameters(self) -> list[Tensor]:
        return [*self.embed.parameters(), self.position,
                *[p for block in self.blocks for p in block.parameters()], *self.head.parameters()]


class NanoGradRunner:
    def __init__(self, model: DLinear, lr: float) -> None:
        self.model, self.optimizer = model, TensorAdam(model.parameters(), lr=lr)

    def _loss(self, inputs, targets, count: int, channels: int):
        prediction = self.model(Tensor(inputs, (count, channels, self.model.lookback)))
        error = prediction - Tensor(targets, (count, channels, self.model.horizon))
        return (error * error).mean()

    def train_step(self, inputs, targets, count: int, channels: int, lr: float) -> float:
        self.optimizer.lr = lr
        self.optimizer.zero_grad()
        loss = self._loss(inputs, targets, count, channels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, inputs, targets, count: int, channels: int) -> float:
        with no_grad():
            return self._loss(inputs, targets, count, channels).item()

    def weights(self):
        return [list(p.data) for p in self.model.parameters()]

    def load(self, weights) -> None:
        for p, values in zip(self.model.parameters(), weights):
            p.data = list(values)


class TorchRunner:
    """DLinear as its authors wrote it (replicate padding, AvgPool1d, two nn.Linear)."""

    def __init__(self, model: DLinear, lr: float, window: int = 25) -> None:
        import torch

        self.torch, self.window = torch, window
        self.seasonal = torch.nn.Linear(model.lookback, model.horizon, dtype=torch.float64)
        self.trend = torch.nn.Linear(model.lookback, model.horizon, dtype=torch.float64)
        with torch.no_grad():
            for ours, theirs in ((model.seasonal, self.seasonal), (model.trend, self.trend)):
                theirs.weight.copy_(torch.tensor(list(ours.weight.data), dtype=torch.float64).reshape(theirs.weight.shape))
                theirs.bias.copy_(torch.tensor(list(ours.bias.data), dtype=torch.float64))
        self.parameters = [*self.seasonal.parameters(), *self.trend.parameters()]
        self.optimizer = torch.optim.Adam(self.parameters, lr=lr)
        self.lookback, self.horizon = model.lookback, model.horizon

    def _forward(self, x):
        torch, half = self.torch, (self.window - 1) // 2
        padded = torch.cat([x[:, :, :1].repeat(1, 1, half), x, x[:, :, -1:].repeat(1, 1, half)], dim=2)
        trend = torch.nn.functional.avg_pool1d(padded, self.window, stride=1)
        return self.seasonal(x - trend) + self.trend(trend)

    def _loss(self, inputs, targets, count, channels):
        torch = self.torch
        x = torch.tensor(inputs, dtype=torch.float64).reshape(count, channels, self.lookback)
        y = torch.tensor(targets, dtype=torch.float64).reshape(count, channels, self.horizon)
        return ((self._forward(x) - y) ** 2).mean()

    def train_step(self, inputs, targets, count, channels, lr) -> float:
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad()
        loss = self._loss(inputs, targets, count, channels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, inputs, targets, count, channels) -> float:
        with self.torch.no_grad():
            return self._loss(inputs, targets, count, channels).item()

    def weights(self):
        return [p.detach().reshape(-1).tolist() for p in self.parameters]

    def load(self, weights) -> None:
        with self.torch.no_grad():
            for p, values in zip(self.parameters, weights):
                p.copy_(self.torch.tensor(values, dtype=self.torch.float64).reshape(p.shape))


def mean_loss(runner, rows, firsts, lookback, horizon, channels) -> float:
    total = 0.0
    for start in range(0, len(firsts), 256):
        chunk = firsts[start : start + 256]
        inputs, targets = batch(rows, chunk, lookback, horizon)
        total += runner.evaluate(inputs, targets, len(chunk), channels) * len(chunk)
    return total / len(firsts)


def run(framework: str, rows, args, horizon: int) -> dict:
    lookback, channels = args.lookback, len(rows[0])
    ranges = splits(rows, lookback, horizon)
    train_firsts, valid_firsts, test_firsts = (windows(rows, *ranges[name], lookback, horizon)
                                               for name in ("train", "validation", "test"))
    if args.model == "patchtst":
        if framework != "nanograd":
            raise SystemExit("--model patchtst runs in NanoGrad only")
        model = PatchTST(lookback, horizon, seed=args.seed)
    else:
        model = DLinear(lookback, horizon, seed=args.seed)
    runner = NanoGradRunner(model, args.lr) if framework == "nanograd" else TorchRunner(model, args.lr)
    rng = random.Random(args.seed)
    best, best_weights, waited, start, steps = math.inf, runner.weights(), 0, perf_counter(), 0
    for epoch in range(1, args.epochs + 1):
        lr = args.lr * 0.5 ** (epoch - 1)                                  # the paper's "type1" schedule
        order = list(train_firsts)
        rng.shuffle(order)
        model.train()
        for first in range(0, len(order) - args.batch_size + 1, args.batch_size):   # drop_last, as theirs
            inputs, targets = batch(rows, order[first : first + args.batch_size], lookback, horizon)
            runner.train_step(inputs, targets, args.batch_size, channels, lr)
            steps += 1
        model.eval()
        valid = mean_loss(runner, rows, valid_firsts, lookback, horizon, channels)
        if valid < best:
            best, best_weights, waited = valid, runner.weights(), 0
        else:
            waited += 1
            if waited >= args.patience:
                break
    runner.load(best_weights)
    seconds = perf_counter() - start
    test = mean_loss(runner, rows, test_firsts, lookback, horizon, channels)
    return {"test_mse": test, "validation_mse": best, "epochs": epoch, "ms_per_step": 1000 * seconds / steps,
            "weights": best_weights}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="data/ett/ETTh1.csv")
    parser.add_argument("--frameworks", nargs="+", choices=("nanograd", "torch"), default=["nanograd"])
    parser.add_argument("--horizons", type=int, nargs="+", default=[96])
    parser.add_argument("--model", choices=("dlinear", "patchtst"), default="dlinear")
    parser.add_argument("--lookback", type=int, default=336)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    rows = standardize(load_ett(args.data), 12 * MONTH)
    results = {}
    for horizon in args.horizons:
        for framework in args.frameworks:
            if framework == "torch":
                import torch

                torch.set_num_threads(1)
            result = run(framework, rows, args, horizon)
            results[(framework, horizon)] = result
            print(f"{framework:8s} horizon {horizon:3d}: test MSE {result['test_mse']:.4f} "
                  f"(validation {result['validation_mse']:.4f}, {result['epochs']} epochs, "
                  f"{result['ms_per_step']:.1f} ms/step)", flush=True)
        if len(args.frameworks) == 2:
            difference = max(abs(a - b) for x, y in zip(results[("nanograd", horizon)]["weights"],
                                                        results[("torch", horizon)]["weights"]) for a, b in zip(x, y))
            print(f"  largest weight difference: {difference:.1e}", flush=True)
    return results


if __name__ == "__main__":
    main()
