"""Run every NanoGrad-vs-PyTorch comparison briefly and print one table.

Each row trains the same model in both frameworks from the same weights on
the same batches (float64, one thread each), then reports the time per
training step and the largest difference between the two sets of weights
afterwards (rounding-level differences mean they computed the same thing).

    .venv/bin/python benchmark_suite.py            # about 10 minutes
    .venv/bin/python benchmark_suite.py --steps 20 # a quick check

Datasets that are not on disk are skipped. The full-length runs and their
accuracies are in docs/benchmarks.md.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
from pathlib import Path

import bert_compare
import mnist_compare
import vae_compare


def image_rows(steps: int, batch_size: int = 32) -> list[dict]:
    rows = []
    for dataset in ("mnist", "fashion", "cifar10"):
        if not Path(mnist_compare.DATASETS[dataset][0]).exists():
            continue
        for model in ("lenet", "resnet", "vit"):
            args = mnist_compare.parse_args([
                "--dataset", dataset, "--model", model, "--train-limit", str(steps * batch_size),
                "--test-limit", "500", "--batch-size", str(batch_size),
            ])
            rows.append(_row(f"{model} on {dataset}", mnist_compare.run, args, per_framework=lambda r: r[0]))
    return rows


def bert_row(steps: int) -> dict:
    args = bert_compare.parse_args(["--steps", str(steps), "--report-every", str(steps), "--eval-batches", "2"])
    return _row("BERT masked characters", bert_compare.run, args, per_framework=lambda r: r)


def vae_row(steps: int) -> dict:
    args = vae_compare.parse_args(["--train-limit", str(steps * 128), "--test-limit", "500"])
    return _row("VAE on mnist", vae_compare.run, args, per_framework=lambda r: r[0])


def _row(name: str, run, args, *, per_framework) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):          # keep the table readable
        results = run(args)
    ours = per_framework(results["nanograd"])["ms_per_step"]
    theirs = per_framework(results["torch"])["ms_per_step"]
    return {"name": name, "nanograd": ours, "torch": theirs, "difference": results["max_weight_difference"]}


def table(rows: list[dict]) -> str:
    lines = [
        "| comparison | NanoGrad ms/step | PyTorch ms/step | ratio | max weight difference |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['nanograd']:.1f} | {row['torch']:.1f} | "
            f"{row['nanograd'] / row['torch']:.1f}x | {row['difference']:.1e} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--steps", type=int, default=100, help="training steps per comparison")
    args = parser.parse_args(argv)
    if importlib.util.find_spec("torch") is None:
        raise SystemExit("benchmark_suite needs PyTorch: python3 -m venv .venv && .venv/bin/pip install torch")

    rows = []
    for make in (lambda: image_rows(args.steps), lambda: [bert_row(args.steps)], lambda: [vae_row(args.steps)]):
        for row in make():
            rows.append(row)
            print(f"done: {row['name']}", flush=True)
    print()
    print(table(rows))


if __name__ == "__main__":
    main()
