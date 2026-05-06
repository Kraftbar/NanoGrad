"""Train a tiny character-level next-token model on built-in text."""

from __future__ import annotations

import argparse
from pathlib import Path

from tensor import Tensor
from tensor_nn import TensorEmbedding, TensorLinear, TensorModule
from text import CharVocab, next_char_dataset, next_char_index_dataset
from train import train_tensor_multiclass_dataset


DEFAULT_TEXT = "hello nanograd\nhello tiny models\n"


class CharBigramModel(TensorModule):
    """Character model: flattened one-hot context -> next-char logits."""

    def __init__(
        self,
        vocab_size: int,
        *,
        context_size: int = 1,
        seed: int = 0,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if context_size <= 0:
            raise ValueError("context_size must be positive")
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.projection = TensorLinear(
            vocab_size * context_size,
            vocab_size,
            seed=seed,
        )

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.projection(inputs)

    def parameters(self) -> list[Tensor]:
        return self.projection.parameters()

    def state_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "context_size": self.context_size,
            "projection": self.projection.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("vocab_size") != self.vocab_size:
            raise ValueError("state vocab_size does not match CharBigramModel")
        if state.get("context_size") != self.context_size:
            raise ValueError("state context_size does not match CharBigramModel")
        self.projection.load_state_dict(state["projection"])


class CharEmbeddingModel(TensorModule):
    """Character model: token-id context -> embeddings -> next-char logits."""

    def __init__(
        self,
        vocab_size: int,
        *,
        context_size: int = 1,
        embedding_dim: int = 16,
        seed: int = 0,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if context_size <= 0:
            raise ValueError("context_size must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.embedding = TensorEmbedding(vocab_size, embedding_dim, seed=seed)
        self.projection = TensorLinear(
            embedding_dim * context_size,
            vocab_size,
            seed=seed + 1,
        )

    def __call__(self, inputs: Tensor) -> Tensor:
        embedded = self.embedding(inputs)
        return self.projection(embedded.flatten(start_axis=1))

    def parameters(self) -> list[Tensor]:
        return [
            *self.embedding.parameters(),
            *self.projection.parameters(),
        ]

    def state_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "context_size": self.context_size,
            "embedding_dim": self.embedding_dim,
            "embedding": self.embedding.state_dict(),
            "projection": self.projection.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("vocab_size") != self.vocab_size:
            raise ValueError("state vocab_size does not match CharEmbeddingModel")
        if state.get("context_size") != self.context_size:
            raise ValueError("state context_size does not match CharEmbeddingModel")
        if state.get("embedding_dim") != self.embedding_dim:
            raise ValueError("state embedding_dim does not match CharEmbeddingModel")
        self.embedding.load_state_dict(state["embedding"])
        self.projection.load_state_dict(state["projection"])


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
) -> str:
    """Generate text by repeatedly taking the highest-logit next character."""

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
        next_index = _argmax(logits.data[: len(vocab)])
        text += vocab.decode([next_index])
    return text


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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
