"""Train a tiny character-level next-token model on built-in text."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from language import CharBigramModel, CharEmbeddingModel
from tensor import Tensor
from tensor_nn import TensorModule
from text import CharVocab, next_char_dataset, next_char_index_dataset
from train import train_tensor_multiclass_dataset


DEFAULT_TEXT = "hello nanograd\nhello tiny models\n"


def run(args: argparse.Namespace) -> None:
    text = load_text(args)
    dataset, vocab = build_dataset(text, args)
    model = build_model(args, vocab_size=len(vocab))
    summary = train_tensor_multiclass_dataset(
        model,
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        shuffle=True,
        seed=args.seed,
    )
    generated = generate_text(
        model,
        vocab,
        seed_text=args.seed_text,
        length=args.generate,
        context_size=args.context_size,
        input_mode=args.model,
        sample_mode=args.sample_mode,
        temperature=args.temperature,
        rng=random.Random(
            args.sample_seed
            if args.sample_seed is not None
            else args.seed
        ),
    )

    print("Character language demo")
    print(f"model:         {args.model}")
    print(f"text source:   {text_source(args)}")
    print(f"text length:   {len(text)}")
    print(f"vocab size:    {len(vocab)}")
    print(f"samples:       {len(dataset)}")
    print(f"context size:  {args.context_size}")
    if args.model == "embedding":
        print(f"embedding dim: {args.embedding_dim}")
    print(f"generation:    {args.sample_mode}")
    if args.sample_mode == "sample":
        print(f"temperature:   {args.temperature}")
    print(f"parameters:    {model.num_parameters()}")
    print(f"initial loss:  {summary.initial_loss:.6f}")
    print(f"final batch:   {summary.final_loss:.6f}")
    if summary.evaluation_loss is not None:
        print(f"train loss:    {summary.evaluation_loss:.6f}")
    print(f"accuracy:      {summary.accuracy:.3f}")
    print(f"runtime:       {summary.elapsed_seconds:.4f}s")
    if summary.examples_per_second is not None:
        print(f"samples/s:     {summary.examples_per_second:.1f}")
    print(f"generated:     {generated!r}")


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
            rng=rng,
        )
        text += vocab.decode([next_index])
    return text


def _select_next_index(
    values: list[float],
    *,
    sample_mode: str,
    temperature: float,
    rng: random.Random | None,
) -> int:
    if sample_mode == "greedy":
        return _argmax(values)
    if sample_mode == "sample":
        return _sample_from_logits(values, temperature=temperature, rng=rng)
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
    rng: random.Random | None = None,
) -> int:
    if not values:
        raise ValueError("sample values must not be empty")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    generator = rng or random.Random()
    row_max = max(values)
    weights = [
        math.exp((value - row_max) / temperature)
        for value in values
    ]
    total = sum(weights)
    threshold = generator.random() * total
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if threshold <= cumulative:
            return index
    return len(values) - 1


def build_dataset(
    text: str,
    args: argparse.Namespace,
):
    if args.model == "bigram":
        return next_char_dataset(text, context_size=args.context_size)
    if args.model == "embedding":
        return next_char_index_dataset(text, context_size=args.context_size)
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
    raise ValueError(f"unknown char model: {args.model}")


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
    parser.add_argument("--model", choices=("bigram", "embedding"), default="bigram")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--max-chars", type=int)
    parser.add_argument("--context-size", type=int, default=1)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-text", default="h")
    parser.add_argument("--generate", type=int, default=32)
    parser.add_argument("--sample-mode", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample-seed", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
