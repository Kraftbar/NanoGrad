"""Train a tiny character-level next-token model on built-in text."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

from language import CharBigramModel, CharEmbeddingModel, CharTransformerModel
from tensor import Tensor
from tensor_nn import TensorModule
from text import (
    CharVocab,
    next_char_dataset,
    next_char_index_dataset,
    next_char_sequence_dataset,
)
from train import (
    train_tensor_multiclass_dataset,
    train_tensor_sequence_multiclass_dataset,
)


DEFAULT_TEXT = "hello nanograd\nhello tiny models\n"
CHECKPOINT_FORMAT = "nanollm.char_demo.v1"
DEFAULT_OPTIONS = {
    "model": "bigram",
    "max_chars": None,
    "context_size": 1,
    "embedding_dim": 16,
    "hidden_dim": None,
    "heads": 1,
    "layers": 1,
    "epochs": 200,
    "batch_size": 8,
    "lr": 0.2,
    "report_every": 0,
    "seed_text": "h",
    "generate": 32,
}
PRESETS = {
    "tiny-transformer": {
        "model": "transformer",
        "max_chars": 128,
        "context_size": 4,
        "embedding_dim": 8,
        "hidden_dim": 16,
        "heads": 1,
        "layers": 1,
        "epochs": 1,
        "batch_size": 4,
        "lr": 0.05,
        "seed_text": "hell",
        "generate": 24,
    },
}


def run(args: argparse.Namespace) -> None:
    text = load_text(args)
    dataset, vocab = build_dataset(text, args)
    model = build_model(args, vocab_size=len(vocab))
    if args.load_model is not None:
        load_checkpoint(
            args.load_model,
            model,
            vocab,
            model_name=args.model,
        )

    train_fn = (
        train_tensor_sequence_multiclass_dataset
        if args.model == "transformer"
        else train_tensor_multiclass_dataset
    )
    summary = train_fn(
        model,
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        shuffle=True,
        seed=args.seed,
        epoch_callback=(
            None
            if args.report_every <= 0
            else lambda epoch, summary: print_epoch_report(
                epoch,
                args.epochs,
                summary,
                report_every=args.report_every,
            )
        ),
    )
    generated = generate_text(
        model,
        vocab,
        seed_text=args.seed_text,
        length=args.generate,
        context_size=args.context_size,
        input_mode=generation_input_mode(args.model),
        sample_mode=args.sample_mode,
        temperature=args.temperature,
        top_k=args.top_k,
        rng=random.Random(
            args.sample_seed
            if args.sample_seed is not None
            else args.seed
        ),
    )

    print("Character language demo")
    if args.preset is not None:
        print(f"preset:        {args.preset}")
    print(f"model:         {args.model}")
    print(f"text source:   {text_source(args)}")
    print(f"text length:   {len(text)}")
    print(f"vocab size:    {len(vocab)}")
    print(f"samples:       {len(dataset)}")
    print(f"context size:  {args.context_size}")
    if args.model in ("embedding", "transformer"):
        print(f"embedding dim: {args.embedding_dim}")
    if args.model == "transformer":
        print(f"hidden dim:    {args.hidden_dim or args.embedding_dim * 4}")
        print(f"heads:         {args.heads}")
        print(f"layers:        {args.layers}")
        print("objective:     sequence")
    print(f"generation:    {args.sample_mode}")
    if args.sample_mode == "sample":
        print(f"temperature:   {args.temperature}")
        if args.top_k is not None:
            print(f"top k:         {args.top_k}")
    print(f"parameters:    {model.num_parameters()}")
    print(f"initial loss:  {summary.initial_loss:.6f}")
    print(f"final batch:   {summary.final_loss:.6f}")
    if summary.evaluation_loss is not None:
        print(f"train loss:    {summary.evaluation_loss:.6f}")
    print(f"accuracy:      {summary.accuracy:.3f}")
    print(f"runtime:       {summary.elapsed_seconds:.4f}s")
    if summary.examples_per_second is not None:
        rate_label = "tokens/s" if args.model == "transformer" else "samples/s"
        print(f"{rate_label}:     {summary.examples_per_second:.1f}")
    print(f"generated:     {generated!r}")
    if args.save_model is not None:
        save_checkpoint(
            args.save_model,
            model,
            vocab,
            model_name=args.model,
        )
        print(f"saved model:   {args.save_model}")


def print_epoch_report(
    epoch: int,
    epochs: int,
    summary,
    *,
    report_every: int,
) -> None:
    if epoch % report_every != 0 and epoch != epochs:
        return

    print(
        f"epoch {epoch}/{epochs} "
        f"loss={_report_loss(summary):.6f} "
        f"accuracy={summary.accuracy:.3f}"
    )


def _report_loss(summary) -> float:
    if summary.evaluation_loss is None:
        return summary.final_loss
    return summary.evaluation_loss


def save_checkpoint(
    path: str | Path,
    model: TensorModule,
    vocab: CharVocab,
    *,
    model_name: str,
) -> None:
    """Save model state plus the vocabulary needed to interpret logits."""

    payload = {
        "format": CHECKPOINT_FORMAT,
        "model": model_name,
        "vocab": vocab.itos,
        "state": model.state_dict(),
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_checkpoint(
    path: str | Path,
    model: TensorModule,
    vocab: CharVocab,
    *,
    model_name: str,
) -> None:
    """Load a char-demo checkpoint, accepting older raw state files."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "format" not in payload:
        model.load_state_dict(payload)
        return
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported char checkpoint format")
    if payload.get("model") != model_name:
        raise ValueError("checkpoint model does not match requested char model")
    if payload.get("vocab") != vocab.itos:
        raise ValueError("checkpoint vocabulary does not match text vocabulary")
    model.load_state_dict(payload["state"])


def generate_text(
    model: TensorModule,
    vocab: CharVocab,
    *,
    seed_text: str,
    length: int,
    context_size: int = 1,
    input_mode: str = "bigram",
    sample_mode: str = "greedy",
    temperature: float = 1.0,
    top_k: int | None = None,
    rng: random.Random | None = None,
) -> str:
    """Generate text from next-character logits."""

    if not seed_text:
        raise ValueError("seed_text must not be empty")
    if length < 0:
        raise ValueError("length must be non-negative")
    if context_size <= 0:
        raise ValueError("context_size must be positive")
    if len(seed_text) < context_size:
        raise ValueError("seed_text must be at least context_size characters")

    text = seed_text
    for _ in range(length):
        context = vocab.encode(text[-context_size:])
        logits = model(_generation_inputs(context, vocab, input_mode=input_mode))
        next_index = _select_next_index(
            logits.data[: len(vocab)],
            sample_mode=sample_mode,
            temperature=temperature,
            top_k=top_k,
            rng=rng,
        )
        text += vocab.decode([next_index])
    return text


def _select_next_index(
    values: list[float],
    *,
    sample_mode: str,
    temperature: float,
    top_k: int | None,
    rng: random.Random | None,
) -> int:
    if sample_mode == "greedy":
        return _argmax(values)
    if sample_mode == "sample":
        return _sample_from_logits(
            values,
            temperature=temperature,
            top_k=top_k,
            rng=rng,
        )
    raise ValueError(f"unknown sample mode: {sample_mode}")


def _argmax(values: list[float]) -> int:
    if not values:
        raise ValueError("argmax values must not be empty")
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def _sample_from_logits(
    values: list[float],
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    rng: random.Random | None = None,
) -> int:
    if not values:
        raise ValueError("sample values must not be empty")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")

    generator = rng or random.Random()
    indexed_values = list(enumerate(values))
    if top_k is not None:
        indexed_values = sorted(
            indexed_values,
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

    row_max = max(value for _index, value in indexed_values)
    weights = [
        math.exp((value - row_max) / temperature)
        for _index, value in indexed_values
    ]
    total = sum(weights)
    threshold = generator.random() * total
    cumulative = 0.0
    for (index, _value), weight in zip(indexed_values, weights):
        cumulative += weight
        if threshold <= cumulative:
            return index
    return indexed_values[-1][0]


def build_dataset(
    text: str,
    args: argparse.Namespace,
):
    if args.model == "bigram":
        return next_char_dataset(text, context_size=args.context_size)
    if args.model == "embedding":
        return next_char_index_dataset(text, context_size=args.context_size)
    if args.model == "transformer":
        return next_char_sequence_dataset(text, context_size=args.context_size)
    raise ValueError(f"unknown char model: {args.model}")


def build_model(args: argparse.Namespace, *, vocab_size: int) -> TensorModule:
    if args.model == "bigram":
        return CharBigramModel(
            vocab_size,
            context_size=args.context_size,
            seed=args.seed,
        )
    if args.model == "embedding":
        return CharEmbeddingModel(
            vocab_size,
            context_size=args.context_size,
            embedding_dim=args.embedding_dim,
            seed=args.seed,
        )
    if args.model == "transformer":
        return CharTransformerModel(
            vocab_size,
            context_size=args.context_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.heads,
            num_layers=args.layers,
            seed=args.seed,
        )
    raise ValueError(f"unknown char model: {args.model}")


def generation_input_mode(model_name: str) -> str:
    if model_name == "bigram":
        return "bigram"
    if model_name in ("embedding", "transformer"):
        return "embedding"
    raise ValueError(f"unknown char model: {model_name}")


def _generation_inputs(
    context: list[int],
    vocab: CharVocab,
    *,
    input_mode: str,
) -> Tensor:
    if input_mode == "bigram":
        return Tensor.from_list([_flatten_one_hot_context(context, vocab)])
    if input_mode == "embedding":
        return Tensor.from_list([context])
    raise ValueError(f"unknown generation input mode: {input_mode}")


def _flatten_one_hot_context(context: list[int], vocab: CharVocab) -> list[float]:
    values = []
    for index in context:
        values.extend(vocab.one_hot(index))
    return values


def load_text(args: argparse.Namespace) -> str:
    if args.text_file is not None:
        text = args.text_file.read_text(encoding="utf-8")
    else:
        text = args.text
    if args.max_chars is not None:
        if args.max_chars <= 0:
            raise ValueError("max_chars must be positive")
        text = text[: args.max_chars]
    return text


def text_source(args: argparse.Namespace) -> str:
    if args.text_file is not None:
        return str(args.text_file)
    return "built-in"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS))
    parser.add_argument(
        "--model",
        choices=("bigram", "embedding", "transformer"),
        default=DEFAULT_OPTIONS["model"],
    )
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_OPTIONS["max_chars"])
    parser.add_argument(
        "--context-size",
        type=int,
        default=DEFAULT_OPTIONS["context_size"],
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=DEFAULT_OPTIONS["embedding_dim"],
    )
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_OPTIONS["hidden_dim"])
    parser.add_argument("--heads", type=int, default=DEFAULT_OPTIONS["heads"])
    parser.add_argument("--layers", type=int, default=DEFAULT_OPTIONS["layers"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_OPTIONS["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_OPTIONS["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_OPTIONS["lr"])
    parser.add_argument(
        "--report-every",
        type=int,
        default=DEFAULT_OPTIONS["report_every"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-text", default=DEFAULT_OPTIONS["seed_text"])
    parser.add_argument("--generate", type=int, default=DEFAULT_OPTIONS["generate"])
    parser.add_argument("--sample-mode", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument("--save-model", type=Path)
    parser.add_argument("--load-model", type=Path)
    explicit_options = _explicit_option_names(
        sys.argv[1:]
        if argv is None
        else argv,
    )
    return apply_preset(parser.parse_args(argv), explicit_options=explicit_options)


def apply_preset(
    args: argparse.Namespace,
    *,
    explicit_options: set[str] | None = None,
) -> argparse.Namespace:
    if args.preset is None:
        return args

    explicit_options = explicit_options or set()
    for name, value in PRESETS[args.preset].items():
        if (
            name not in explicit_options
            and getattr(args, name) == DEFAULT_OPTIONS[name]
        ):
            setattr(args, name, value)
    return args


def _explicit_option_names(argv: list[str]) -> set[str]:
    names = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        name = token.split("=", 1)[0]
        names.add(name.removeprefix("--").replace("-", "_"))
    return names


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
