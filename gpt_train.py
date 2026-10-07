"""Train a character-level GPT on a text file, nanoGPT style.

Where ``char_demo.py`` walks the dataset in epochs, this script follows
nanoGPT's ``train.py``: a fixed number of steps, each on a batch of random
windows from the training split, with periodic evaluation on both splits,
a warmup + cosine learning rate, gradient clipping, AdamW-style weight
decay, and the best checkpoint kept by validation loss.

The ``baby`` preset is nanoGPT's CPU recipe for Tiny Shakespeare
(README: "a baby GPT ... gets us a loss of only 1.88"):

    python3 gpt_train.py --text-file data/tinyshakespeare/input.txt \\
        --preset baby --workers 4 --out-dir runs/baby

Checkpoints use char_checkpoint's format, so they can be sampled later:

    python3 char_demo.py --load-model runs/baby/model.json --generate-only \\
        --sample-mode sample --top-k 10 --seed-text "ROMEO:" --generate 300
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

from char_checkpoint import load_checkpoint, save_checkpoint
from char_demo import generate_text
from language import CharTransformerModel
from optim import TensorAdam
from parallel import DataParallel, sequence_loss
from schedules import warmup_cosine
from tensor import no_grad
from text import CharVocab
from train import evaluation_mode


PRESETS = {
    # nanoGPT README, "I only have a MacBook" / CPU command.
    "baby": dict(
        layers=4, heads=4, embedding_dim=128, context_size=64, batch_size=12,
        steps=2000, lr=1e-3, min_lr=1e-4, warmup_steps=100, dropout=0.0,
    ),
    # A smaller rung that trains in minutes, for trying things out.
    "mini": dict(
        layers=2, heads=2, embedding_dim=32, context_size=32, batch_size=16,
        steps=500, lr=3e-3, min_lr=3e-4, warmup_steps=50, dropout=0.0,
    ),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def split_text(text: str, validation_fraction: float) -> tuple[str, str]:
    """First 90% for training, last 10% for validation (nanoGPT's split)."""

    cut = int(len(text) * (1.0 - validation_fraction))
    return text[:cut], text[cut:]


def random_batch(
    encoded: list[int],
    *,
    batch_size: int,
    context_size: int,
    rng: random.Random,
) -> tuple[list[list[int]], list[list[int]]]:
    """``batch_size`` random windows and the same windows shifted by one."""

    starts = [rng.randrange(len(encoded) - context_size) for _ in range(batch_size)]
    xs = [encoded[start : start + context_size] for start in starts]
    ys = [encoded[start + 1 : start + context_size + 1] for start in starts]
    return xs, ys


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def build_model(vocab_size: int, config: argparse.Namespace) -> CharTransformerModel:
    return CharTransformerModel(
        vocab_size,
        context_size=config.context_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=4 * config.embedding_dim,
        num_heads=config.heads,
        num_layers=config.layers,
        feed_forward_activation="gelu",
        tie_embeddings=True,
        init="gpt2",
        dropout=config.dropout,
        seed=config.seed,
    )


def estimate_loss(model, splits: dict[str, list[int]], args, rng: random.Random) -> dict[str, float]:
    """Mean loss over ``eval_batches`` random batches per split (eval mode)."""

    losses = {}
    with no_grad(), evaluation_mode(model):
        for name, encoded in splits.items():
            total = 0.0
            for _ in range(args.eval_batches):
                xs, ys = random_batch(
                    encoded,
                    batch_size=args.batch_size,
                    context_size=args.context_size,
                    rng=rng,
                )
                total += sequence_loss(model, xs, ys).item()
            losses[name] = total / args.eval_batches
    return losses


def train(args: argparse.Namespace) -> dict:
    text = Path(args.text_file).read_text(encoding="utf-8")
    if args.max_chars is not None:
        text = text[: args.max_chars]
    vocab = CharVocab.from_text(text)
    train_text, validation_text = split_text(text, args.validation_fraction)
    splits = {"train": vocab.encode(train_text), "val": vocab.encode(validation_text)}

    model = build_model(len(vocab), args)
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
        if args.resume and (out_dir / "model.json").exists():
            load_checkpoint(out_dir / "model.json", model, vocab, model_name="transformer")

    optimizer = TensorAdam(
        model.parameters(),
        lr=args.lr,
        beta1=0.9,
        beta2=0.99,
        weight_decay=args.weight_decay,
        weight_decay_min_ndim=2,
        max_grad_norm=args.max_grad_norm,
    )
    schedule = warmup_cosine(
        max_lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=min(args.warmup_steps, args.steps - 1),   # short test runs
        decay_steps=args.steps,
    )

    print(f"vocab {len(vocab)}, train {len(splits['train'])} chars, val {len(splits['val'])} chars")
    print(f"model {model.num_parameters()} parameters, workers {args.workers}")

    data_rng = random.Random(args.seed)
    eval_rng = random.Random(args.seed + 1)
    log_rows: list[dict] = []
    best_val = float("inf")
    start = perf_counter()
    parallel_context = (
        DataParallel(model, sequence_loss, workers=args.workers) if args.workers > 1 else nullcontext()
    )
    with parallel_context as parallel:
        for step in range(args.steps + 1):
            if step % args.eval_every == 0 or step == args.steps:
                losses = estimate_loss(model, splits, args, eval_rng)
                elapsed = perf_counter() - start
                row = {"step": step, "train_loss": losses["train"], "val_loss": losses["val"],
                       "lr": schedule(step), "seconds": elapsed}
                log_rows.append(row)
                print(f"step {step:5d}: train {losses['train']:.4f}, val {losses['val']:.4f}, "
                      f"{elapsed / 60:.1f} min", flush=True)
                if out_dir is not None:
                    _write_log(out_dir / "log.csv", log_rows)
                    if losses["val"] < best_val:
                        best_val = losses["val"]
                        save_checkpoint(out_dir / "model.json", model, vocab, model_name="transformer")
            if step == args.steps:
                break

            xs, ys = random_batch(
                splits["train"],
                batch_size=args.batch_size,
                context_size=args.context_size,
                rng=data_rng,
            )
            optimizer.lr = schedule(step)
            optimizer.zero_grad()
            if parallel is not None:
                parallel.loss_and_gradients(xs, ys)
            else:
                model.train()
                sequence_loss(model, xs, ys).backward()
            optimizer.step()

    samples = [
        generate_text(
            model,
            vocab,
            seed_text=args.seed_text,
            length=args.generate,
            context_size=args.context_size,
            input_mode="embedding",
            sample_mode="sample",
            temperature=args.temperature,
            top_k=args.top_k,
            rng=random.Random(args.seed + 100 + index),
        )
        for index in range(args.num_samples)
    ]
    for sample in samples:
        print("-" * 60)
        print(sample)
    if out_dir is not None:
        (out_dir / "samples.txt").write_text(("\n" + "-" * 60 + "\n").join(samples) + "\n")
    return {"log": log_rows, "samples": samples, "best_val": best_val}


def _write_log(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="mini")
    parser.add_argument("--max-chars", type=int)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    for name in ("layers", "heads", "embedding_dim", "context_size", "batch_size", "steps", "warmup_steps"):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    for name in ("lr", "min_lr", "dropout"):
        parser.add_argument("--" + name.replace("_", "-"), type=float)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed-text", default="\n")
    parser.add_argument("--generate", type=int, default=300)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args(argv)
    for name, value in PRESETS[args.preset].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


def main(argv: list[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
