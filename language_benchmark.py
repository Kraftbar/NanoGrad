"""Small language-model timing checks for NanoGrad."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from time import perf_counter

from char_demo import PRESETS
from language import CharTransformerModel
from losses import softmax_cross_entropy
from tensor import Tensor


BENCHMARK_PRESETS = ("tiny-gpt", "lite-gpt", "small-gpt")


def benchmark_transformer(config: dict, *, batch_size: int = 4) -> None:
    vocab_size = 16
    context_size = config["context_size"]
    model = CharTransformerModel(
        vocab_size=vocab_size,
        context_size=context_size,
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_heads=config["heads"],
        num_layers=config["layers"],
        feed_forward_activation=config["activation"],
        tie_embeddings=config["tie_embeddings"],
        seed=0,
    )
    rows = [
        [
            (batch * context_size + position) % vocab_size
            for position in range(context_size)
        ]
        for batch in range(batch_size)
    ]
    inputs = Tensor.from_list(rows)
    targets = Tensor.from_list([
        [
            (token + 1) % vocab_size
            for token in row
        ]
        for row in rows
    ])

    logits = model.sequence_logits(inputs)
    batch_size, context_size, classes = logits.shape
    loss = softmax_cross_entropy(
        logits.reshape((batch_size * context_size, classes)),
        targets.reshape((targets.numel,)),
    )
    model.zero_grad()
    loss.backward()


def run(args: argparse.Namespace) -> None:
    benchmarks = [
        (
            f"{args.preset} transformer fwd+bwd",
            lambda: benchmark_transformer(
                PRESETS[args.preset],
                batch_size=args.batch_size,
            ),
        ),
    ]

    print("NanoGrad language benchmarks")
    print(f"preset: {args.preset}")
    print(f"batch size: {args.batch_size}")
    print(f"repeat: {args.repeat}")
    for name, function in benchmarks:
        seconds = average_seconds(function, repeat=args.repeat)
        print(f"{name}: {seconds:.6f}s")


def average_seconds(function: Callable[[], None], *, repeat: int) -> float:
    if repeat <= 0:
        raise ValueError("repeat must be positive")

    start = perf_counter()
    for _ in range(repeat):
        function()
    return (perf_counter() - start) / repeat


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=BENCHMARK_PRESETS, default="tiny-gpt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
