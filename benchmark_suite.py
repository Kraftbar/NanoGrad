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


def family_rows(steps: int) -> list[dict]:
    """Every newer family, briefly, through its own script's --frameworks nanograd torch.
    Each entry: (name, module, data that must exist, arguments, ms per step from the results)."""

    import adversarial, audio, fewshot, flow, kws, linkpred, recommend, segment, sr, tagger

    both = ["--frameworks", "nanograd", "torch"]
    families = [
        ("NeuMF on MovieLens-1M", recommend, "data/ml-1m",
         both + ["--epochs", "1", "--step-limit", str(steps), "--limit-users", "300"],
         lambda r: r["ms_per_step"]),
        ("VGAE on Cora", linkpred, "data/cora",
         both + ["--epochs", str(steps), "--log-every", str(steps)],
         lambda r: 1000 * r["seconds"] / steps),
        ("MAF on MNIST", flow, "data/mnist",
         both + ["--epochs", "1", "--train-limit", str(steps * 20), "--test-limit", "200", "--hidden", "256",
                 "--layers", "3", "--calibration", "500"],
         lambda r: r["ms_per_step"]),
        ("SRCNN on 91-image", sr, "data/sr",
         both + ["--epochs", "1", "--patch-limit", str(steps * 16), "--test-sets", "set5"],
         lambda r: r["ms_per_step"]),
        ("prototypical net on Omniglot", fewshot, "data/omniglot/omniglot28.pkl",
         both + ["--episodes", str(max(1, steps // 5)), "--train-way", "20", "--test-episodes", "50",
                 "--log-every", str(steps)],
         lambda r: r["ms_per_episode"]),
        ("FGSM training on MNIST", adversarial, "data/mnist",
         both + ["--epochs", "1", "--train-limit", str(steps * 100), "--test-limit", "100", "--pgd-steps", "5",
                 "--restarts", "1"],
         lambda r: 1000 * r["train_seconds"] / steps),
        ("CNN on spoken digits", audio, "data/fsdd/recordings",
         both + ["--epochs", "1"],
         lambda r: r["ms_per_step"]),
        ("DS-CNN keyword spotting", kws, "data/speech_commands/mfcc.pkl",
         both + ["--epochs", "1", "--train-limit", str(steps * 100), "--test-limit", "500"],
         lambda r: r["ms_per_step"]),
        ("BiLSTM-CRF tagger on UD EWT", tagger, "data/ud_ewt/en_ewt-ud-train.conllu",
         both + ["--epochs", "1", "--train-limit", str(steps * 32), "--dropout", "0"],
         lambda r: r["ms_per_step"]),
        ("U-Net on CamVid", segment, "data/camvid/camvid_176x240.pkl",
         both + ["--epochs", "1", "--train-limit", str(max(2, steps // 10) * 4), "--test-limit", "4"],
         lambda r: r["ms_per_step"]),
    ]
    rows = []
    for name, module, data, argv, ms in families:
        if not Path(data).exists():
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            results = module.main(argv)
        rows.append({"name": name, "nanograd": ms(results["nanograd"]), "torch": ms(results["torch"]),
                     "difference": results.get("max_weight_difference", float("nan"))})
    return rows


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
    for make in (lambda: image_rows(args.steps), lambda: [bert_row(args.steps)], lambda: [vae_row(args.steps)],
                 lambda: family_rows(args.steps)):
        for row in make():
            rows.append(row)
            print(f"done: {row['name']}", flush=True)
    print()
    print(table(rows))


if __name__ == "__main__":
    main()
