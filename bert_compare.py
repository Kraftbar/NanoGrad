"""Train the tiny BERT encoder on Tiny Shakespeare in NanoGrad and PyTorch.

The task is masked character modeling (see bert.py): hide 15% of the
characters in 64-character windows and predict them from both sides. Both
frameworks start from the same weights (PyTorch runs the same architecture
via torch_twin.py) and see the same masked batches, so in float64 they
should produce the same losses. The script reports masked-character
accuracy on held-out text and the time per step.

    .venv/bin/python bert_compare.py --steps 500
    python3 bert_compare.py --frameworks nanograd
"""

from __future__ import annotations

import argparse
import importlib.util
import random
from pathlib import Path
from time import perf_counter

from bert import TinyBERT, mask_tokens, masked_loss
from gpt_train import random_batch, split_text
from optim import TensorAdam
from tensor import no_grad
from text import CharVocab
from train import evaluation_mode


FRAMEWORKS = ("nanograd", "torch")


def make_batches(encoded, count, args, rng, model) -> list[tuple]:
    """``count`` masked batches: (inputs, positions, targets)."""

    batches = []
    for _ in range(count):
        windows, _ = random_batch(encoded, batch_size=args.batch_size, context_size=args.context_size, rng=rng)
        batches.append(mask_tokens(windows, vocab_size=model.vocab_size, mask_id=model.mask_id, rng=rng))
    return batches


class NanoGradRunner:
    def __init__(self, model: TinyBERT, lr: float) -> None:
        self.model = model
        self.optimizer = TensorAdam(model.parameters(), lr=lr)

    def train_step(self, inputs, positions, targets) -> float:
        self.optimizer.zero_grad()
        loss = masked_loss(self.model, inputs, positions, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, inputs, positions) -> list[int]:
        with no_grad(), evaluation_mode(self.model):
            logits = self.model(inputs)
        vocab = logits.shape[-1]
        return [
            max(range(vocab), key=lambda k, p=position: logits.data[p * vocab + k])
            for position in positions
        ]

    def weights(self) -> list:
        return [parameter.data for parameter in self.model.parameters()]


class TorchRunner:
    def __init__(self, model: TinyBERT, lr: float, threads: int) -> None:
        import torch

        from torch_twin import TorchTwin

        torch.set_num_threads(threads)
        self.torch = torch
        self.twin = TorchTwin(model)
        self.optimizer = torch.optim.Adam(self.twin.parameters, lr=lr)

    def _scores(self, inputs, positions):
        logits = self.twin(self.torch.tensor(inputs))
        return logits.reshape(-1, logits.shape[-1])[self.torch.tensor(positions)]

    def train_step(self, inputs, positions, targets) -> float:
        self.optimizer.zero_grad()
        loss = self.torch.nn.functional.cross_entropy(self._scores(inputs, positions), self.torch.tensor(targets))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, inputs, positions) -> list[int]:
        with self.torch.no_grad():
            return self._scores(inputs, positions).argmax(dim=1).tolist()

    def weights(self) -> list:
        return [parameter.detach().reshape(-1).tolist() for parameter in self.twin.parameters]


def masked_accuracy(runner, batches) -> float:
    correct = total = 0
    for inputs, positions, targets in batches:
        predictions = runner.predict(inputs, positions)
        correct += sum(p == t for p, t in zip(predictions, targets))
        total += len(targets)
    return correct / total


def run(args: argparse.Namespace) -> dict:
    text = Path(args.text_file).read_text(encoding="utf-8")
    vocab = CharVocab.from_text(text)
    train_text, validation_text = split_text(text, 0.1)
    train_encoded, validation_encoded = vocab.encode(train_text), vocab.encode(validation_text)

    model = TinyBERT(
        len(vocab), context_size=args.context_size, embedding_dim=args.embedding_dim,
        num_heads=args.heads, num_layers=args.layers, seed=args.seed,
    )
    print(f"vocab {len(vocab)} + [MASK], model {model.num_parameters()} parameters")
    rng = random.Random(args.seed)
    train_batches = make_batches(train_encoded, args.steps, args, rng, model)
    validation_batches = make_batches(validation_encoded, args.eval_batches, args, random.Random(args.seed + 1), model)

    runners = {}
    for name in args.frameworks:
        if name == "nanograd":
            runners[name] = NanoGradRunner(model, args.lr)
        else:
            if importlib.util.find_spec("torch") is None:
                raise SystemExit("PyTorch is not installed; use --frameworks nanograd")
            runners[name] = TorchRunner(model, args.lr, args.torch_threads)   # copies the initial weights

    results = {}
    for name, runner in runners.items():
        start = perf_counter()
        losses = []
        for step, batch in enumerate(train_batches, start=1):
            losses.append(runner.train_step(*batch))
            if step % args.report_every == 0 or step == args.steps:
                recent = losses[-args.report_every :]
                print(f"{name:8s} step {step:5d}: train loss {sum(recent) / len(recent):.4f}, "
                      f"{perf_counter() - start:.1f} s", flush=True)
        seconds = perf_counter() - start
        accuracy = masked_accuracy(runner, validation_batches)
        results[name] = {"seconds": seconds, "ms_per_step": 1000 * seconds / args.steps, "accuracy": accuracy}
        print(f"{name:8s} masked-character accuracy on held-out text: {100 * accuracy:.2f}%, "
              f"{1000 * seconds / args.steps:.1f} ms/step")

    if len(runners) == 2:
        difference = max(
            abs(a - b)
            for left, right in zip(runners["nanograd"].weights(), runners["torch"].weights())
            for a, b in zip(left, right)
        )
        print(f"max |weight difference| after training: {difference:.3e}")
        results["max_weight_difference"] = difference
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--text-file", default="data/tinyshakespeare/input.txt")
    parser.add_argument("--frameworks", nargs="+", choices=FRAMEWORKS, default=list(FRAMEWORKS))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
