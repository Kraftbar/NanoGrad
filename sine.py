"""Predicting sine waves with an LSTM trained by L-BFGS (PyTorch's time_sequence_prediction).

100 sine waves of period 2 * pi * 20 samples, 1,000 samples each, every
one shifted by a random phase. A 2-layer LSTM (51 units each, then a
linear read-out) learns to predict each next sample from the samples so
far, on 97 waves; 3 are kept to test. Then it extends the test waves 1,000
samples into the future, feeding its own predictions back in.

Training is full-batch L-BFGS (lr 0.8, 15 steps of up to 20 iterations),
as in the example, not stochastic gradient descent. Both frameworks run
the same model from the same weights.

    python3 sine.py
    .venv/bin/python sine.py --frameworks nanograd torch
"""

from __future__ import annotations

import argparse
import math
import random
from time import perf_counter

from optim import TensorLBFGS
from rnn import LSTM
from tensor import Tensor, no_grad
from tensor_nn import TensorLinear, TensorModule


def sine_waves(count: int = 100, length: int = 1000, period: float = 20.0, seed: int = 2) -> list[list[float]]:
    rng = random.Random(seed)
    return [[math.sin((step + shift) / period) for step in range(length)]
            for shift in (rng.randrange(-4 * int(period), 4 * int(period)) for _ in range(count))]


class SineModel(TensorModule):
    def __init__(self, hidden: int = 51, seed: int = 0) -> None:
        self.lstm1, self.lstm2 = LSTM(1, hidden, seed=seed), LSTM(hidden, hidden, seed=seed + 1)
        self.readout = TensorLinear(hidden, 1, seed=seed + 2)

    def __call__(self, waves: Tensor, future: int = 0) -> Tensor:
        """Predictions (B, T + future): the next sample after each input sample,
        then ``future`` more, each fed back in as the next input."""

        batch, length = waves.shape
        out1, state1 = self.lstm1.run(waves.reshape((batch, length, 1)))
        out2, state2 = self.lstm2.run(out1, None)
        outputs = self.readout(out2).reshape((batch, length))
        if not future:
            return outputs
        predictions = [outputs.data[row * length + length - 1] for row in range(batch)]
        extended = []
        for _ in range(future):
            step = Tensor(predictions, (batch, 1, 1))
            hidden1, state1 = self.lstm1.run(step, state1)
            hidden2, state2 = self.lstm2.run(hidden1, state2)
            predictions = list(self.readout(hidden2).data)
            extended.append(predictions)
        future_data = [extended[t][row] for row in range(batch) for t in range(future)]
        joined = [value for row in range(batch) for value in
                  list(outputs.data[row * length : (row + 1) * length]) + future_data[row * future : (row + 1) * future]]
        return Tensor(joined, (batch, length + future))

    def parameters(self) -> list[Tensor]:
        return [*self.lstm1.parameters(), *self.lstm2.parameters(), *self.readout.parameters()]


def mse(predictions: Tensor, targets: Tensor) -> Tensor:
    error = predictions - targets
    return (error * error).mean()


def run_nanograd(model: SineModel, train_x, train_y, test_x, test_y, args) -> dict:
    optimizer = TensorLBFGS(model.parameters(), lr=args.lr)
    history, start = [], perf_counter()

    def closure() -> float:
        for parameter in model.parameters():
            parameter.zero_grad()
        loss = mse(model(train_x), train_y)
        loss.backward()
        return loss.item()

    for step in range(1, args.steps + 1):
        train_loss = optimizer.step(closure)
        with no_grad():
            test_loss = mse(model(test_x), test_y).item()
        history.append((train_loss, test_loss))
        print(f"  step {step:2d}: train loss {train_loss:.3e}, test loss {test_loss:.3e}", flush=True)
    return {"history": history, "seconds": perf_counter() - start}


def run_torch(model: SineModel, train_x, train_y, test_x, test_y, args) -> dict:
    import torch

    torch.set_num_threads(1)

    def t(values) -> "torch.Tensor":
        return torch.tensor(list(values.data), dtype=torch.float64).reshape(values.shape)

    lstm = torch.nn.LSTM(1, args.hidden, num_layers=2, batch_first=True, dtype=torch.float64)
    readout = torch.nn.Linear(args.hidden, 1, dtype=torch.float64)
    with torch.no_grad():
        for index, layer in enumerate((model.lstm1, model.lstm2)):
            getattr(lstm, f"weight_ih_l{index}").copy_(t(layer.input_weight))
            getattr(lstm, f"weight_hh_l{index}").copy_(t(layer.hidden_weight))
            getattr(lstm, f"bias_ih_l{index}").copy_(t(layer.bias))
            getattr(lstm, f"bias_hh_l{index}").zero_()
            getattr(lstm, f"bias_hh_l{index}").requires_grad_(False)     # NanoGrad's LSTM has one bias
        readout.weight.copy_(t(model.readout.weight))
        readout.bias.copy_(t(model.readout.bias))
    parameters = [p for p in (*lstm.parameters(), *readout.parameters()) if p.requires_grad]
    optimizer = torch.optim.LBFGS(parameters, lr=args.lr)
    x, y, tx, ty = (t(v) for v in (train_x, train_y, test_x, test_y))
    history, start = [], perf_counter()

    def closure():
        optimizer.zero_grad()
        out, _ = lstm(x.unsqueeze(-1))
        loss = ((readout(out).squeeze(-1) - y) ** 2).mean()
        loss.backward()
        return loss

    for step in range(1, args.steps + 1):
        train_loss = optimizer.step(closure).item()
        with torch.no_grad():
            out, _ = lstm(tx.unsqueeze(-1))
            test_loss = ((readout(out).squeeze(-1) - ty) ** 2).mean().item()
        history.append((train_loss, test_loss))
        print(f"  step {step:2d}: train loss {train_loss:.3e}, test loss {test_loss:.3e}", flush=True)
    return {"history": history, "seconds": perf_counter() - start}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--frameworks", nargs="+", choices=("nanograd", "torch"), default=["nanograd"])
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--hidden", type=int, default=51)
    parser.add_argument("--lr", type=float, default=0.8)
    parser.add_argument("--waves", type=int, default=100)
    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--future", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    waves = sine_waves(args.waves, args.length)
    test, train = waves[:3], waves[3:]

    def pair(rows):                                  # inputs: samples 0..T-2, targets: 1..T-1
        return (Tensor([v for row in rows for v in row[:-1]], (len(rows), args.length - 1)),
                Tensor([v for row in rows for v in row[1:]], (len(rows), args.length - 1)))

    (train_x, train_y), (test_x, test_y) = pair(train), pair(test)
    results = {}
    for framework in args.frameworks:
        print(framework, flush=True)
        model = SineModel(args.hidden, seed=args.seed)
        runner = run_nanograd if framework == "nanograd" else run_torch
        results[framework] = runner(model, train_x, train_y, test_x, test_y, args)
        final = results[framework]["history"][-1]
        print(f"{framework:8s}: test loss {final[1]:.3e} after {args.steps} L-BFGS steps, "
              f"{results[framework]['seconds']:.0f} s", flush=True)
        if framework == "nanograd" and args.future:
            # Extend the test waves on the model's own predictions, against the true continuation.
            with no_grad():
                extended = model(test_x, future=args.future)
            truth = [row[args.length : args.length + args.future]
                     for row in sine_waves(args.waves, args.length + args.future)[:3]]
            width = extended.shape[1]
            errors = [(extended.data[row * width + args.length - 1 + t] - truth[row][t]) ** 2
                      for row in range(3) for t in range(args.future)]
            results["future_mse"] = sum(errors) / len(errors)
            print(f"  free-running prediction of the next {args.future} samples: "
                  f"mean squared error {results['future_mse']:.3e}", flush=True)
    return results


if __name__ == "__main__":
    main()
