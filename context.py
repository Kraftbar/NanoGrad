"""Longer context than training: RoPE extrapolation vs position interpolation.

A model with rotary positions (RoPE) has no position table, so it can
technically read sequences longer than it was trained on. It reads them
badly: positions past the training length rotate queries and keys by
angles the model never saw, and attention scores there are out of
distribution. Position interpolation (Chen et al., 2023, used to extend
LLaMA from 2K to 32K tokens) instead squeezes the longer sequence into
the trained range. Position p becomes p * (trained length / new length),
so every angle is one the model knows, only finer-grained. The model then
needs a short fine-tune to get used to the closer spacing.

The experiment, on the baby GPT with LLaMA parts (trained at 64
characters of context):

    trained length     the model as trained, windows of 64
    extrapolate        windows of 128 / 256 with the raw positions
    interpolate        positions scaled by 1/2 / 1/4, no training
    interpolate + FT   the same, fine-tuned for a few hundred steps at the new length

Loss is in nats per character on the validation text, over every position
and over the last quarter of each window (the positions that could
benefit from more context).

    python3 context.py --run runs/baby_llama
"""

from __future__ import annotations

import argparse
import copy
import random
from time import perf_counter

from gpt_train import load_run, split_text
from losses import softmax_cross_entropy
from optim import TensorAdam
from tensor import Tensor, no_grad
from text import CharVocab
from train import evaluation_mode


def window_losses(model, tokens: list[int], length: int, windows: int) -> tuple[float, float]:
    """Mean next-token loss over every position, and over the last quarter, of
    ``windows`` evenly spaced windows of ``length + 1`` tokens."""

    step = (len(tokens) - length - 1) // windows
    total = tail = 0.0
    quarter = length - length // 4
    with no_grad(), evaluation_mode(model):
        for w in range(windows):
            chunk = tokens[w * step : w * step + length + 1]
            logits = model.sequence_logits(Tensor.from_list([chunk[:-1]]))
            vocab = logits.shape[-1]
            flat = logits.reshape((length, vocab))
            total += softmax_cross_entropy(flat, chunk[1:]).item()
            tail += softmax_cross_entropy(flat[quarter:], chunk[1 + quarter :]).item()
    return total / windows, tail / windows


def fine_tune(model, tokens: list[int], length: int, args) -> None:
    rng = random.Random(args.seed)
    optimizer = TensorAdam(model.parameters(), lr=args.lr)
    model.train()
    for step in range(1, args.steps + 1):
        starts = [rng.randrange(len(tokens) - length - 1) for _ in range(args.batch_size)]
        inputs = [tokens[s : s + length] for s in starts]
        targets = [t for s in starts for t in tokens[s + 1 : s + length + 1]]
        optimizer.zero_grad()
        logits = model.sequence_logits(Tensor.from_list(inputs))
        loss = softmax_cross_entropy(logits.reshape((len(targets), logits.shape[-1])), targets)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"    fine-tune step {step}: loss {loss.item():.4f}", flush=True)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", default="runs/baby_llama", help="a gpt_train.py --out-dir trained with RoPE")
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--windows", type=int, default=40, help="validation windows per measurement")
    parser.add_argument("--steps", type=int, default=300, help="fine-tuning steps at each new length")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    base, validation, config = load_run(args.run)
    text = open(config.text_file, encoding="utf-8").read()
    train_text, _ = split_text(text, config.validation_fraction)
    training = CharVocab.from_text(text).encode(train_text)
    trained = base.context_size
    results = {}

    def report(name: str, losses: tuple[float, float]) -> None:
        results[name] = losses
        print(f"{name:34s} loss {losses[0]:.4f}, last quarter {losses[1]:.4f}", flush=True)

    report(f"trained length ({trained})", window_losses(base, validation, trained, args.windows))
    for length in args.lengths:
        start = perf_counter()
        extrapolated = copy.deepcopy(base)
        extrapolated.extend_context(length, interpolate=False)
        report(f"{length}: extrapolate", window_losses(extrapolated, validation, length, args.windows))
        interpolated = copy.deepcopy(base)
        interpolated.extend_context(length, interpolate=True)
        report(f"{length}: interpolate", window_losses(interpolated, validation, length, args.windows))
        fine_tune(interpolated, training, length, args)
        report(f"{length}: interpolate + {args.steps} steps",
               window_losses(interpolated, validation, length, args.windows))
        tuned = copy.deepcopy(base)                          # the control: fine-tuning alone, raw positions
        tuned.extend_context(length, interpolate=False)
        fine_tune(tuned, training, length, args)
        report(f"{length}: extrapolate + {args.steps} steps", window_losses(tuned, validation, length, args.windows))
        print(f"  ({perf_counter() - start:.0f} s)", flush=True)
    return results


if __name__ == "__main__":
    main()
