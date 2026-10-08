"""Uncertainty: MC dropout regression on the UCI benchmark, NanoGrad vs PyTorch.

A regression network usually gives one number. Gal and Ghahramani (2016,
"Dropout as a Bayesian Approximation") showed that a network trained with
dropout can also say how unsure it is: keep dropout on at test time and
run the input T times. Each run drops different units, i.e. uses a
different thinned network, a sample from an approximate posterior over
weights. The mean of the T predictions is the prediction; their spread
(plus the noise level 1/tau) is the uncertainty. With y_t the T outputs,
the predictive log-likelihood of the true y is

    log p(y | x) = log (1/T) sum_t N(y; y_t, 1/tau)
                 = logsumexp_t(-tau/2 (y - y_t)^2) - log T - log(2 pi)/2 + log(tau)/2

The benchmark (Hernandez-Lobato and Adams, 2015) is 9 small UCI tables,
each with fixed random 90/10 train/test splits (20; 5 for protein). Per
split, as in Gal's code: hold out the last 20% of the training rows,
grid-search the dropout rate and tau on them by log-likelihood, retrain on
all training rows with the best pair, and report test RMSE (of the MC
mean) and log-likelihood. The network: dropout on the inputs, one ReLU
layer of 50 units (100 for protein) with dropout, a linear output; inputs
and targets standardized; Adam (lr 0.001, batch 128) on mean squared error
plus the weight decay l^2 (1 - p) / (2 N tau) * sum W^2 with l = 0.01;
40 epochs (20 for protein) times ``--epoch-multiplier``.

The paper reports (40 epochs; RMSE / log-likelihood): Boston 2.97 / -2.46,
concrete 5.23 / -3.04, energy 1.66 / -1.99, kin8nm 0.10 / 0.95,
naval 0.01 / 3.80, power 4.02 / -2.80, protein 4.36 / -2.89,
wine 0.62 / -0.93, yacht 1.11 / -1.55.

    python3 uci.py --dataset concrete
    python3 uci.py --dataset all --splits 5
    .venv/bin/python uci.py --dataset yacht --frameworks nanograd torch

Data: data/uci/<dataset>/ (see data/uci/README.md).
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from time import perf_counter

from optim import TensorAdam
from tensor import Tensor, no_grad
from tensor_nn import TensorDropout, TensorLinear, TensorModule


DATASETS = ("bostonHousing", "concrete", "energy", "kin8nm", "naval-propulsion-plant", "power-plant",
            "protein-tertiary-structure", "wine-quality-red", "yacht")
LENGTHSCALE, BATCH = 1e-2, 128


class Dataset:
    """One UCI table with its splits and hyperparameter grid."""

    def __init__(self, folder: str | Path) -> None:
        folder = Path(folder)

        def numbers(name: str) -> list[float]:
            return [float(v) for v in (folder / name).read_text().split()]

        rows = [[float(v) for v in line.split()] for line in (folder / "data.txt").read_text().splitlines()
                if line.strip()]
        features, target = [int(i) for i in numbers("index_features.txt")], int(numbers("index_target.txt")[0])
        self.x = [[row[i] for i in features] for row in rows]
        self.y = [row[target] for row in rows]
        self.splits = int(numbers("n_splits.txt")[0])
        self.hidden = int(numbers("n_hidden.txt")[0])
        self.epochs = int(numbers("n_epochs.txt")[0])
        self.dropout_rates = numbers("dropout_rates.txt")
        self.taus = numbers("tau_values.txt")
        self.folder = folder

    def split(self, index: int) -> tuple[list[int], list[int]]:
        def ids(name: str) -> list[int]:
            return [int(float(v)) for v in (self.folder / name).read_text().split()]

        return ids(f"index_train_{index}.txt"), ids(f"index_test_{index}.txt")


class Standardizer:
    """Zero mean, unit standard deviation per column, from the training rows."""

    def __init__(self, rows: list[list[float]]) -> None:
        count = len(rows)
        self.mean = [sum(column) / count for column in zip(*rows)]
        spread = [math.sqrt(sum((v - m) ** 2 for v in column) / count) for column, m in zip(zip(*rows), self.mean)]
        self.std = [s if s > 0 else 1.0 for s in spread]

    def __call__(self, rows: list[list[float]]) -> list[list[float]]:
        return [[(v - m) / s for v, m, s in zip(row, self.mean, self.std)] for row in rows]


class DropoutNet(TensorModule):
    """inputs -> dropout -> ReLU layer -> dropout -> linear output."""

    def __init__(self, inputs: int, hidden: int, dropout: float, seed: int = 0) -> None:
        self.input_dropout = TensorDropout(dropout, seed=seed)
        self.hidden = TensorLinear(inputs, hidden, seed=seed + 1)
        self.hidden_dropout = TensorDropout(dropout, seed=seed + 2)
        self.output = TensorLinear(hidden, 1, seed=seed + 3)

    def __call__(self, x: Tensor) -> Tensor:
        return self.output(self.hidden_dropout(self.hidden(self.input_dropout(x)).relu()))

    def parameters(self) -> list[Tensor]:
        return [*self.hidden.parameters(), *self.output.parameters()]


class NanoGradRunner:
    def __init__(self, inputs: int, hidden: int, dropout: float, decay: float, seed: int) -> None:
        self.model = DropoutNet(inputs, hidden, dropout, seed)
        self.optimizer = TensorAdam(self.model.parameters(), lr=0.001)
        self.decay = decay

    def train_step(self, x: list[list[float]], y: list[float]) -> None:
        self.optimizer.zero_grad()
        error = self.model(Tensor([v for row in x for v in row], (len(x), len(x[0])))) - Tensor(y, (len(y), 1))
        loss = (error * error).mean()
        for weight in (self.model.hidden.weight, self.model.output.weight):    # Keras' l2(decay) on the weights
            loss = loss + (weight * weight).sum() * self.decay
        loss.backward()
        self.optimizer.step()

    def sample(self, x: list[list[float]], copies: int) -> list[float]:
        """``copies`` stochastic predictions of every row (dropout stays on), copy-major."""

        flat = [v for row in x for v in row] * copies
        with no_grad():
            return list(self.model(Tensor(flat, (len(x) * copies, len(x[0])))).data)


class TorchRunner:
    def __init__(self, inputs: int, hidden: int, dropout: float, decay: float, seed: int) -> None:
        import torch

        torch.manual_seed(seed)
        self.torch = torch
        self.model = torch.nn.Sequential(torch.nn.Dropout(dropout), torch.nn.Linear(inputs, hidden), torch.nn.ReLU(),
                                         torch.nn.Dropout(dropout), torch.nn.Linear(hidden, 1)).double()
        for layer in (self.model[1], self.model[4]):            # Glorot uniform and zero bias, as in Keras
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.decay = decay

    def train_step(self, x, y) -> None:
        torch = self.torch
        self.optimizer.zero_grad()
        error = self.model(torch.tensor(x, dtype=torch.float64)) - torch.tensor(y, dtype=torch.float64)[:, None]
        loss = (error ** 2).mean() + self.decay * sum((layer.weight ** 2).sum() for layer in (self.model[1], self.model[4]))
        loss.backward()
        self.optimizer.step()

    def sample(self, x, copies: int) -> list[float]:
        torch = self.torch
        with torch.no_grad():
            return self.model(torch.tensor(x * copies, dtype=torch.float64)).reshape(-1).tolist()


def fit(make_runner, x, y, *, hidden: int, dropout: float, tau: float, epochs: int, seed: int):
    """Train one network on (x, y); returns the runner and the input and target scalers."""

    inputs = Standardizer(x)
    mean = sum(y) / len(y)
    std = math.sqrt(sum((v - mean) ** 2 for v in y) / len(y))
    xs, ys = inputs(x), [(v - mean) / std for v in y]
    decay = LENGTHSCALE ** 2 * (1 - dropout) / (2.0 * len(x) * tau)
    runner = make_runner(len(x[0]), hidden, dropout, decay, seed)
    rng = random.Random(seed)
    order = list(range(len(xs)))
    for _ in range(epochs):
        rng.shuffle(order)
        for first in range(0, len(order), BATCH):
            batch = order[first : first + BATCH]
            runner.train_step([xs[i] for i in batch], [ys[i] for i in batch])
    return runner, inputs, (mean, std)


def evaluate(runner, inputs: Standardizer, target: tuple[float, float], x, y, tau: float, samples: int):
    """Test RMSE of the MC mean, and the MC predictive log-likelihood (in the data's units)."""

    mean, std = target
    xs, count = inputs(x), len(x)
    totals = [0.0] * count
    # log sum_t exp(-tau/2 (y - y_t)^2), accumulated over chunks of samples with a running maximum.
    peak, scaled = [-math.inf] * count, [0.0] * count
    per_chunk = max(1, min(samples, 200_000 // count))
    done = 0
    while done < samples:
        copies = min(per_chunk, samples - done)
        outputs = runner.sample(xs, copies)
        for c in range(copies):
            for i in range(count):
                value = outputs[c * count + i] * std + mean
                totals[i] += value
                exponent = -0.5 * tau * (y[i] - value) ** 2
                if exponent > peak[i]:
                    scaled[i] = scaled[i] * math.exp(peak[i] - exponent) + 1.0
                    peak[i] = exponent
                else:
                    scaled[i] += math.exp(exponent - peak[i])
        done += copies
    rmse = math.sqrt(sum((y[i] - totals[i] / samples) ** 2 for i in range(count)) / count)
    lls = [peak[i] + math.log(scaled[i]) - math.log(samples) - 0.5 * math.log(2 * math.pi) + 0.5 * math.log(tau)
           for i in range(count)]
    return rmse, sum(lls) / count


def run_split(dataset: Dataset, index: int, make_runner, args) -> dict:
    train_ids, test_ids = dataset.split(index)
    x, y = [dataset.x[i] for i in train_ids], [dataset.y[i] for i in train_ids]
    epochs = dataset.epochs * args.epoch_multiplier
    cut = int(0.8 * len(x))
    best = None
    for dropout in dataset.dropout_rates:                 # grid search on the last 20% of the training rows
        for tau in dataset.taus:
            runner, inputs, target = fit(make_runner, x[:cut], y[:cut], hidden=dataset.hidden, dropout=dropout,
                                         tau=tau, epochs=epochs, seed=args.seed + index)
            _, ll = evaluate(runner, inputs, target, x[cut:], y[cut:], tau, args.search_samples)
            if best is None or ll > best[0]:
                best = (ll, dropout, tau)
    _, dropout, tau = best
    runner, inputs, target = fit(make_runner, x, y, hidden=dataset.hidden, dropout=dropout, tau=tau, epochs=epochs,
                                 seed=args.seed + index)
    rmse, ll = evaluate(runner, inputs, target, [dataset.x[i] for i in test_ids], [dataset.y[i] for i in test_ids],
                        tau, args.samples)
    return {"rmse": rmse, "ll": ll, "dropout": dropout, "tau": tau}


def mean_and_error(values: list[float]) -> tuple[float, float]:
    """Mean and standard error, as the paper reports them."""

    mean = sum(values) / len(values)
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) / math.sqrt(len(values))


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="data/uci")
    parser.add_argument("--dataset", nargs="+", default=["concrete"], help=f"names, or all: {', '.join(DATASETS)}")
    parser.add_argument("--frameworks", nargs="+", choices=("nanograd", "torch"), default=["nanograd"])
    parser.add_argument("--splits", type=int, help="use only the first N splits")
    parser.add_argument("--epoch-multiplier", type=int, default=1)
    parser.add_argument("--samples", type=int, default=10_000, help="MC dropout samples at test time")
    parser.add_argument("--search-samples", type=int, default=1_000, help="MC samples during the grid search")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    names = DATASETS if args.dataset == ["all"] else args.dataset
    results = {}
    for name in names:
        dataset = Dataset(Path(args.data) / name)
        splits = min(dataset.splits, args.splits or dataset.splits)
        for framework in args.frameworks:
            if framework == "torch":
                import torch

                torch.set_num_threads(1)
            make_runner = TorchRunner if framework == "torch" else NanoGradRunner
            start, rows = perf_counter(), []
            for index in range(splits):
                rows.append(run_split(dataset, index, make_runner, args))
                print(f"  {name} {framework} split {index:2d}: RMSE {rows[-1]['rmse']:.3f}, "
                      f"log-likelihood {rows[-1]['ll']:.3f} (dropout {rows[-1]['dropout']}, tau {rows[-1]['tau']})",
                      flush=True)
            rmse, ll = mean_and_error([r["rmse"] for r in rows]), mean_and_error([r["ll"] for r in rows])
            results[(name, framework)] = {"rmse": rmse, "ll": ll, "splits": rows}
            print(f"{name:28s} {framework:8s}: RMSE {rmse[0]:.3f} +- {rmse[1]:.3f}, "
                  f"log-likelihood {ll[0]:.3f} +- {ll[1]:.3f} over {splits} splits, "
                  f"{perf_counter() - start:.0f} s", flush=True)
    return results


if __name__ == "__main__":
    main()
