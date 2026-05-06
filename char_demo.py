"""Train a tiny character-level next-token model on built-in text."""

from __future__ import annotations

import argparse

from tensor import Tensor
from tensor_nn import TensorLinear, TensorModule
from text import CharVocab, next_char_dataset
from train import train_tensor_multiclass_dataset


DEFAULT_TEXT = "hello nanograd\nhello tiny models\n"


class CharBigramModel(TensorModule):
    """One-step character model: one-hot current char -> next-char logits."""

    def __init__(self, vocab_size: int, *, seed: int = 0) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.vocab_size = vocab_size
        self.projection = TensorLinear(vocab_size, vocab_size, seed=seed)

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.projection(inputs)

    def parameters(self) -> list[Tensor]:
        return self.projection.parameters()

    def state_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "projection": self.projection.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("vocab_size") != self.vocab_size:
            raise ValueError("state vocab_size does not match CharBigramModel")
        self.projection.load_state_dict(state["projection"])


def run(args: argparse.Namespace) -> None:
    dataset, vocab = next_char_dataset(args.text)
    model = CharBigramModel(len(vocab), seed=args.seed)
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
    )

    print("Character bigram demo")
    print(f"text length:   {len(args.text)}")
    print(f"vocab size:    {len(vocab)}")
    print(f"samples:       {len(dataset)}")
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
    model: CharBigramModel,
    vocab: CharVocab,
    *,
    seed_text: str,
    length: int,
) -> str:
    """Generate text by repeatedly taking the highest-logit next character."""

    if not seed_text:
        raise ValueError("seed_text must not be empty")
    if length < 0:
        raise ValueError("length must be non-negative")

    text = seed_text
    current = seed_text[-1]
    for _ in range(length):
        index = vocab.encode(current)[0]
        logits = model(Tensor.from_list([vocab.one_hot(index)]))
        next_index = _argmax(logits.data[: len(vocab)])
        current = vocab.decode([next_index])
        text += current
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=DEFAULT_TEXT)
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
