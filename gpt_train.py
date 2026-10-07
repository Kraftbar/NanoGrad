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

``--tokenizer bpe`` trains a byte-pair-encoding tokenizer (bpe.py) on the
training split first, so each token covers about two characters. Losses per
token are then not comparable with a character model's, so the log also
reports validation *bits per character*, which are.

Character-model checkpoints use char_checkpoint's format, so they can be
sampled later:

    python3 char_demo.py --load-model runs/baby/model.json --generate-only \\
        --sample-mode sample --top-k 10 --seed-text "ROMEO:" --generate 300
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import kernels
from bpe import BPETokenizer
from char_checkpoint import load_checkpoint, save_checkpoint
from char_demo import generate_text
from language import CharTransformerModel
from optim import TensorAdam
from parallel import DataParallel, sequence_loss
from rnn import CharLSTMModel
from sampling import select_next_index
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
    # The same size and recipe with LLaMA's parts: RMSNorm, rotary position
    # embeddings and a SwiGLU feed-forward (hidden width 8/3 * embedding, as
    # LLaMA does, so the parameter count stays about the same).
    "baby-llama": dict(
        layers=4, heads=4, embedding_dim=128, context_size=64, batch_size=12,
        steps=2000, lr=1e-3, min_lr=1e-4, warmup_steps=100, dropout=0.0,
        norm="rms", positions="rope", activation="swiglu",
    ),
    # Karpathy's char-rnn in the same recipe: a 2-layer LSTM of about the
    # same size (embedding 64, hidden 256: 874,881 parameters).
    "baby-lstm": dict(
        architecture="lstm", layers=2, embedding_dim=64, lstm_hidden=256, context_size=64,
        batch_size=12, steps=2000, lr=1e-3, min_lr=1e-4, warmup_steps=100, dropout=0.0,
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


def build_model(vocab_size: int, config: argparse.Namespace):
    if config.architecture in ("lstm", "gru"):
        return CharLSTMModel(
            vocab_size, embedding_dim=config.embedding_dim, hidden=config.lstm_hidden,
            num_layers=config.layers, cell=config.architecture, seed=config.seed,
        )
    hidden_dim = 4 * config.embedding_dim
    if config.activation == "swiglu":
        hidden_dim = -(-8 * config.embedding_dim // 3 // 8) * 8   # 8/3 * C, rounded up to 8
    return CharTransformerModel(
        vocab_size,
        context_size=config.context_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=hidden_dim,
        num_heads=config.heads,
        num_layers=config.layers,
        feed_forward_activation=config.activation,
        norm=config.norm,
        positions=config.positions,
        tie_embeddings=True,
        init="gpt2",
        dropout=config.dropout,
        seed=config.seed,
    )


def estimate_loss(
    model,
    splits: dict[str, list[int]],
    args,
    rng: random.Random,
    parallel: DataParallel | None = None,
) -> dict[str, float]:
    """Mean loss over ``eval_batches`` random batches per split (eval mode)."""

    losses = {}
    for name, encoded in splits.items():
        total = 0.0
        for _ in range(args.eval_batches):
            xs, ys = random_batch(
                encoded,
                batch_size=args.batch_size,
                context_size=args.context_size,
                rng=rng,
            )
            if parallel is not None:
                total += parallel.evaluate(xs, ys)
            else:
                with no_grad(), evaluation_mode(model):
                    total += sequence_loss(model, xs, ys).item()
        losses[name] = total / args.eval_batches
    return losses


def build_tokenizer(train_text: str, text: str, args: argparse.Namespace):
    """Characters, or a BPE vocabulary learned from the training split only."""

    if args.tokenizer == "bpe":
        return BPETokenizer.train(train_text, args.bpe_vocab_size)
    return CharVocab.from_text(text)


def generate_tokens(model, prompt: list[int], *, length: int, context_size: int,
                    temperature: float, top_k: int | None, rng: random.Random,
                    kv_cache: bool = True) -> list[int]:
    """Sample ``length`` tokens after ``prompt``, one at a time.

    Without the KV cache, every new token runs the model on the last
    ``context_size`` tokens (as nanoGPT's generate does). With it, each
    token only adds one row of work (see ``CharTransformerModel.step_logits``).
    Learned positions cannot slide, so when the window fills up the cache
    restarts from the most recent half window: the model then sees between
    half and all of a window of history, instead of always a full one.
    """

    def choose(logits) -> int:
        return select_next_index(list(logits.data), sample_mode="sample",
                                 temperature=temperature, top_k=top_k, rng=rng)

    if not prompt:
        raise ValueError("generation needs at least one prompt token")
    tokens = list(prompt)
    if not kv_cache:
        for _ in range(length):
            with no_grad(), evaluation_mode(model):
                logits = model([tokens[-context_size:]])
            tokens.append(choose(logits))
        return tokens

    with no_grad(), evaluation_mode(model):
        cache, fed = model.new_cache(), 0
        window_start = max(0, len(tokens) - context_size)
        for _ in range(length):
            if len(tokens) - window_start > context_size:
                window_start = len(tokens) - context_size // 2
                cache, fed = model.new_cache(), 0
            while window_start + fed < len(tokens):        # feed what the cache lacks
                logits = model.step_logits(tokens[window_start + fed], fed, cache)
                fed += 1
            tokens.append(choose(logits))
    return tokens


def train(args: argparse.Namespace) -> dict:
    kernels.set_default_dtype(args.dtype)
    text = Path(args.text_file).read_text(encoding="utf-8")
    if args.max_chars is not None:
        text = text[: args.max_chars]
    train_text, validation_text = split_text(text, args.validation_fraction)
    vocab = build_tokenizer(train_text, text, args)
    splits = {"train": vocab.encode(train_text), "val": vocab.encode(validation_text)}
    # A token covers this many characters on average, so loss per character
    # = loss per token / chars_per_token (1 for a character model).
    chars_per_token = len(validation_text) / len(splits["val"])

    model = build_model(len(vocab), args)
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
        if args.tokenizer == "bpe":
            vocab.save(out_dir / "tokenizer.json")
        if args.resume and (out_dir / "model.json").exists():
            if not _char_checkpoint(args):
                model.load(out_dir / "model.json")
            else:
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

    print(f"{args.tokenizer} vocab {len(vocab)}, train {len(splits['train'])} tokens, "
          f"val {len(splits['val'])} tokens ({chars_per_token:.2f} chars/token)")
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
                losses = estimate_loss(model, splits, args, eval_rng, parallel)
                elapsed = perf_counter() - start
                bits_per_char = losses["val"] / math.log(2) / chars_per_token
                row = {"step": step, "train_loss": losses["train"], "val_loss": losses["val"],
                       "val_bits_per_char": bits_per_char, "lr": schedule(step), "seconds": elapsed}
                log_rows.append(row)
                print(f"step {step:5d}: train {losses['train']:.4f}, val {losses['val']:.4f} "
                      f"({bits_per_char:.3f} bits/char), {elapsed / 60:.1f} min", flush=True)
                if out_dir is not None:
                    _write_log(out_dir / "log.csv", log_rows)
                    if losses["val"] < best_val:
                        best_val = losses["val"]
                        if not _char_checkpoint(args):
                            model.save(out_dir / "model.json")
                        else:
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

    if _char_checkpoint(args):
        samples = [_char_sample(model, vocab, args, index) for index in range(args.num_samples)]
    else:
        samples = [
            vocab.decode(generate_tokens(
                model, vocab.encode(args.seed_text), length=args.generate, context_size=args.context_size,
                temperature=args.temperature, top_k=args.top_k, rng=random.Random(args.seed + 100 + index),
                kv_cache=args.architecture == "transformer",
            ))
            for index in range(args.num_samples)
        ]
    for sample in samples:
        print("-" * 60)
        print(sample)
    if out_dir is not None:
        (out_dir / "samples.txt").write_text(("\n" + "-" * 60 + "\n").join(samples) + "\n")
    return {"log": log_rows, "samples": samples, "best_val": best_val}


def _char_checkpoint(args: argparse.Namespace) -> bool:
    """Character transformers use char_checkpoint's format (char_demo can load
    them); other models save their state plus, for BPE, tokenizer.json."""

    return args.tokenizer == "char" and args.architecture == "transformer"


def _char_sample(model, vocab, args, index: int) -> str:
    return generate_text(
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
    parser.add_argument("--tokenizer", choices=("char", "bpe"), default="char")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--bpe-vocab-size", type=int, default=512)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    for name in ("layers", "heads", "embedding_dim", "context_size", "batch_size", "steps", "warmup_steps"):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    for name in ("lr", "min_lr", "dropout"):
        parser.add_argument("--" + name.replace("_", "-"), type=float)
    parser.add_argument("--architecture", choices=("transformer", "lstm", "gru"))
    parser.add_argument("--lstm-hidden", type=int)
    parser.add_argument("--norm", choices=("layer", "rms"))
    parser.add_argument("--positions", choices=("learned", "rope"))
    parser.add_argument("--activation", choices=("gelu", "relu", "swiglu"))
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
    defaults = {"architecture": "transformer", "lstm_hidden": 256,
                "norm": "layer", "positions": "learned", "activation": "gelu"}   # GPT-2's parts
    for name, value in {**defaults, **PRESETS[args.preset]}.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


def main(argv: list[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
