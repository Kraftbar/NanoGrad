"""Train a tiny character-level next-token model on built-in text."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

from language import CharBigramModel, CharEmbeddingModel, CharTransformerModel
from metrics import perplexity
from tensor import Tensor
from tensor_nn import TensorModule
from text import (
    CharVocab,
    mean_distinct_ngram_ratio,
    next_char_dataset,
    next_char_index_dataset,
    next_char_sequence_dataset,
)
from train import (
    TrainingSummary,
    evaluate_tensor_multiclass_dataset,
    evaluate_tensor_sequence_multiclass_dataset,
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
    "activation": "relu",
    "tie_embeddings": False,
    "eval_only": False,
    "epochs": 200,
    "batch_size": 8,
    "lr": 0.2,
    "no_shuffle": False,
    "optimizer": "sgd",
    "weight_decay": 0.0,
    "weight_decay_min_ndim": None,
    "report_every": 0,
    "validation_chars": 0,
    "max_grad_norm": None,
    "metrics_file": None,
    "seed_text": "h",
    "seed_file": None,
    "generate": 32,
    "num_samples": 1,
    "generate_only": False,
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
        "activation": "relu",
        "tie_embeddings": False,
        "epochs": 1,
        "batch_size": 4,
        "lr": 0.05,
        "seed_text": "hell",
        "generate": 24,
    },
    "tiny-gpt": {
        "model": "transformer",
        "max_chars": 128,
        "context_size": 4,
        "embedding_dim": 8,
        "hidden_dim": 16,
        "heads": 1,
        "layers": 1,
        "activation": "gelu",
        "tie_embeddings": True,
        "epochs": 1,
        "batch_size": 4,
        "lr": 0.01,
        "optimizer": "adam",
        "weight_decay": 0.01,
        "weight_decay_min_ndim": 2,
        "max_grad_norm": 1.0,
        "seed_text": "hell",
        "generate": 24,
    },
    "small-gpt": {
        "model": "transformer",
        "max_chars": 512,
        "context_size": 8,
        "embedding_dim": 16,
        "hidden_dim": 64,
        "heads": 2,
        "layers": 2,
        "activation": "gelu",
        "tie_embeddings": True,
        "epochs": 1,
        "batch_size": 8,
        "lr": 0.005,
        "optimizer": "adam",
        "weight_decay": 0.01,
        "weight_decay_min_ndim": 2,
        "max_grad_norm": 1.0,
        "seed_text": "hello na",
        "generate": 80,
    },
}


def run(args: argparse.Namespace) -> None:
    if args.load_model is not None:
        apply_checkpoint_config(args, args.load_model)
    if args.generate_only:
        run_generate_only(args)
        return

    text = load_text(args)
    train_text, validation_text = split_train_validation_text(text, args)
    base_vocab = CharVocab.from_text(text) if validation_text is not None else None
    dataset, vocab = build_dataset(train_text, args, vocab=base_vocab)
    validation_dataset = (
        None
        if validation_text is None
        else build_dataset(validation_text, args, vocab=vocab)[0]
    )
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
    evaluate_fn = (
        evaluate_tensor_sequence_multiclass_dataset
        if args.model == "transformer"
        else evaluate_tensor_multiclass_dataset
    )
    epoch_records = []
    if args.eval_only:
        summary = evaluate_only_summary(
            model,
            dataset,
            validation_dataset=validation_dataset,
            evaluate_fn=evaluate_fn,
            batch_size=args.batch_size,
        )
        if args.metrics_file is not None:
            epoch_records.append(epoch_record(0, summary))
    else:
        summary = train_fn(
            model,
            dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            shuffle=not args.no_shuffle,
            seed=args.seed,
            optimizer_name=args.optimizer,
            weight_decay=args.weight_decay,
            weight_decay_min_ndim=args.weight_decay_min_ndim,
            max_grad_norm=args.max_grad_norm,
            validation_dataset=validation_dataset,
            epoch_callback=epoch_callback(
                args,
                epoch_records,
            ),
        )
    generated_samples = generate_samples(
        model,
        vocab,
        args,
    )

    print("Character language demo")
    if args.preset is not None:
        print(f"preset:        {args.preset}")
    print(f"model:         {args.model}")
    if args.eval_only:
        print("eval only:     True")
    print(f"text source:   {text_source(args)}")
    print(f"text length:   {len(text)}")
    print(f"vocab size:    {len(vocab)}")
    print(f"samples:       {len(dataset)}")
    if validation_dataset is not None:
        print(f"val samples:   {len(validation_dataset)}")
    print(f"context size:  {args.context_size}")
    if args.model in ("embedding", "transformer"):
        print(f"embedding dim: {args.embedding_dim}")
    if args.model == "transformer":
        print(f"hidden dim:    {args.hidden_dim or args.embedding_dim * 4}")
        print(f"heads:         {args.heads}")
        print(f"layers:        {args.layers}")
        print(f"activation:    {args.activation}")
        print(f"tie embeddings: {args.tie_embeddings}")
        print("objective:     sequence")
    if not args.eval_only and args.max_grad_norm is not None:
        print(f"max grad norm: {args.max_grad_norm}")
    if not args.eval_only:
        print(f"optimizer:     {args.optimizer}")
        print(f"shuffle:       {not args.no_shuffle}")
        if args.weight_decay:
            print(f"weight decay:  {args.weight_decay}")
            if args.weight_decay_min_ndim is not None:
                print(f"decay ndim >=: {args.weight_decay_min_ndim}")
    print(f"generation:    {args.sample_mode}")
    if args.num_samples != 1:
        print(f"num samples:   {args.num_samples}")
    if args.sample_mode == "sample":
        print(f"temperature:   {args.temperature}")
        if args.top_k is not None:
            print(f"top k:         {args.top_k}")
    print(f"parameters:    {model.num_parameters()}")
    print(f"initial loss:  {summary.initial_loss:.6f}")
    if not args.eval_only:
        print(f"final batch:   {summary.final_loss:.6f}")
    if summary.evaluation_loss is not None:
        print(f"train loss:    {summary.evaluation_loss:.6f}")
        print(f"train ppl:     {perplexity(summary.evaluation_loss):.3f}")
    print(f"accuracy:      {summary.accuracy:.3f}")
    if summary.validation_loss is not None:
        print(f"val loss:      {summary.validation_loss:.6f}")
        print(f"val ppl:       {perplexity(summary.validation_loss):.3f}")
    if summary.validation_accuracy is not None:
        print(f"val accuracy:  {summary.validation_accuracy:.3f}")
    print(f"runtime:       {summary.elapsed_seconds:.4f}s")
    if summary.examples_per_second is not None:
        rate_label = "tokens/s" if args.model == "transformer" else "samples/s"
        print(f"{rate_label}:     {summary.examples_per_second:.1f}")
    print_generated_samples(generated_samples)
    if args.save_model is not None:
        save_checkpoint(
            args.save_model,
            model,
            vocab,
            model_name=args.model,
        )
        print(f"saved model:   {args.save_model}")
    if args.metrics_file is not None:
        write_epoch_metrics(args.metrics_file, epoch_records)
        print(f"metrics file:  {args.metrics_file}")


def run_generate_only(args: argparse.Namespace) -> None:
    if args.load_model is None:
        raise ValueError("generate-only requires --load-model")
    if args.metrics_file is not None:
        raise ValueError("generate-only does not produce epoch metrics")

    vocab = checkpoint_vocab(args.load_model)
    model = build_model(args, vocab_size=len(vocab))
    load_checkpoint(
        args.load_model,
        model,
        vocab,
        model_name=args.model,
    )
    generated_samples = generate_samples(
        model,
        vocab,
        args,
    )

    print("Character language demo")
    if args.preset is not None:
        print(f"preset:        {args.preset}")
    print(f"model:         {args.model}")
    print("generate only: True")
    print(f"checkpoint:    {args.load_model}")
    print(f"vocab size:    {len(vocab)}")
    print(f"context size:  {args.context_size}")
    if args.model in ("embedding", "transformer"):
        print(f"embedding dim: {args.embedding_dim}")
    if args.model == "transformer":
        print(f"hidden dim:    {args.hidden_dim or args.embedding_dim * 4}")
        print(f"heads:         {args.heads}")
        print(f"layers:        {args.layers}")
        print(f"activation:    {args.activation}")
        print(f"tie embeddings: {args.tie_embeddings}")
    print(f"generation:    {args.sample_mode}")
    if args.num_samples != 1:
        print(f"num samples:   {args.num_samples}")
    if args.sample_mode == "sample":
        print(f"temperature:   {args.temperature}")
        if args.top_k is not None:
            print(f"top k:         {args.top_k}")
    print(f"parameters:    {model.num_parameters()}")
    print_generated_samples(generated_samples)
    if args.save_model is not None:
        save_checkpoint(
            args.save_model,
            model,
            vocab,
            model_name=args.model,
        )
        print(f"saved model:   {args.save_model}")


def epoch_callback(args: argparse.Namespace, records: list[dict]):
    if args.report_every <= 0 and args.metrics_file is None:
        return None

    def callback(epoch: int, summary) -> None:
        if args.metrics_file is not None:
            records.append(epoch_record(epoch, summary))
        if args.report_every > 0:
            print_epoch_report(
                epoch,
                args.epochs,
                summary,
                report_every=args.report_every,
            )

    return callback


def epoch_record(epoch: int, summary) -> dict:
    return {
        "epoch": epoch,
        "loss": _report_loss(summary),
        "perplexity": perplexity(_report_loss(summary)),
        "accuracy": summary.accuracy,
        "val_loss": summary.validation_loss,
        "val_perplexity": (
            None
            if summary.validation_loss is None
            else perplexity(summary.validation_loss)
        ),
        "val_accuracy": summary.validation_accuracy,
        "elapsed_seconds": summary.elapsed_seconds,
        "examples_seen": summary.examples_seen,
    }


def write_epoch_metrics(path: str | Path, records: list[dict]) -> None:
    fieldnames = [
        "epoch",
        "loss",
        "perplexity",
        "accuracy",
        "val_loss",
        "val_perplexity",
        "val_accuracy",
        "elapsed_seconds",
        "examples_seen",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def print_epoch_report(
    epoch: int,
    epochs: int,
    summary,
    *,
    report_every: int,
) -> None:
    if epoch % report_every != 0 and epoch != epochs:
        return

    message = (
        f"epoch {epoch}/{epochs} "
        f"loss={_report_loss(summary):.6f} "
        f"ppl={perplexity(_report_loss(summary)):.3f} "
        f"accuracy={summary.accuracy:.3f}"
    )
    if summary.validation_loss is not None:
        message += f" val_loss={summary.validation_loss:.6f}"
        message += f" val_ppl={perplexity(summary.validation_loss):.3f}"
    if summary.validation_accuracy is not None:
        message += f" val_accuracy={summary.validation_accuracy:.3f}"
    print(message)


def _report_loss(summary) -> float:
    if summary.evaluation_loss is None:
        return summary.final_loss
    return summary.evaluation_loss


def evaluate_only_summary(
    model: TensorModule,
    dataset,
    *,
    validation_dataset=None,
    evaluate_fn,
    batch_size: int,
) -> TrainingSummary:
    train_eval = evaluate_fn(
        model,
        dataset,
        batch_size=batch_size,
    )
    validation_eval = (
        None
        if validation_dataset is None
        else evaluate_fn(
            model,
            validation_dataset,
            batch_size=batch_size,
        )
    )
    elapsed_seconds = train_eval.elapsed_seconds
    if validation_eval is not None:
        elapsed_seconds += validation_eval.elapsed_seconds

    return TrainingSummary(
        history=[train_eval.loss],
        elapsed_seconds=elapsed_seconds,
        accuracy=train_eval.accuracy,
        validation_accuracy=(
            None
            if validation_eval is None
            else validation_eval.accuracy
        ),
        evaluation_loss=train_eval.loss,
        validation_loss=(
            None
            if validation_eval is None
            else validation_eval.loss
        ),
        examples_seen=train_eval.examples_seen,
        confusion_matrix=train_eval.confusion_matrix,
        validation_confusion_matrix=(
            None
            if validation_eval is None
            else validation_eval.confusion_matrix
        ),
    )


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


def apply_checkpoint_config(args: argparse.Namespace, path: str | Path) -> None:
    """Fill unspecified CLI model settings from a v1 checkpoint."""

    payload = _checkpoint_payload(path)
    if "format" not in payload:
        return
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported char checkpoint format")

    state = payload.get("state")
    if not isinstance(state, dict):
        raise ValueError("checkpoint state must be a dictionary")

    updates = _checkpoint_config_updates(payload.get("model"), state)
    missing = [
        name
        for name, value in updates.items()
        if value is None
    ]
    if missing:
        raise ValueError(
            "checkpoint is missing model config: " + ", ".join(missing)
        )

    explicit_options = getattr(args, "_explicit_options", set())
    for name, value in updates.items():
        if name not in explicit_options:
            setattr(args, name, value)


def checkpoint_vocab(path: str | Path) -> CharVocab:
    """Load the saved character vocabulary from a v1 checkpoint."""

    payload = _checkpoint_payload(path)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("generate-only requires a v1 char checkpoint")
    vocab = payload.get("vocab")
    if not isinstance(vocab, list):
        raise ValueError("checkpoint vocabulary must be a list")
    return CharVocab(vocab)


def _checkpoint_payload(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _checkpoint_config_updates(model_name, state: dict) -> dict:
    if model_name == "bigram":
        return {
            "model": "bigram",
            "context_size": state.get("context_size"),
        }
    if model_name == "embedding":
        return {
            "model": "embedding",
            "context_size": state.get("context_size"),
            "embedding_dim": state.get("embedding_dim"),
        }
    if model_name == "transformer":
        return {
            "model": "transformer",
            "context_size": state.get("context_size"),
            "embedding_dim": state.get("embedding_dim"),
            "hidden_dim": state.get("hidden_dim"),
            "heads": state.get("num_heads", 1),
            "layers": state.get("num_layers"),
            "activation": state.get("feed_forward_activation", "relu"),
            "tie_embeddings": state.get("tie_embeddings", False),
        }
    raise ValueError("checkpoint model is unknown")


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


def generate_samples(
    model: TensorModule,
    vocab: CharVocab,
    args: argparse.Namespace,
) -> list[str]:
    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive")

    seed_text = generation_seed_text(args)
    rng = random.Random(
        args.sample_seed
        if args.sample_seed is not None
        else args.seed
    )
    return [
        generate_text(
            model,
            vocab,
            seed_text=seed_text,
            length=args.generate,
            context_size=args.context_size,
            input_mode=generation_input_mode(args.model),
            sample_mode=args.sample_mode,
            temperature=args.temperature,
            top_k=args.top_k,
            rng=rng,
        )
        for _ in range(args.num_samples)
    ]


def generation_seed_text(args: argparse.Namespace) -> str:
    if args.seed_file is None:
        return args.seed_text

    text = args.seed_file.read_text(encoding="utf-8")
    if not text:
        raise ValueError("seed_file must not be empty")
    return text


def print_generated_samples(samples: list[str]) -> None:
    print(f"sample dist-2: {mean_distinct_ngram_ratio(samples, n=2):.3f}")
    if len(samples) == 1:
        print(f"generated:     {samples[0]!r}")
        return
    for index, sample in enumerate(samples, start=1):
        print(f"generated {index}:   {sample!r}")


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
    *,
    vocab: CharVocab | None = None,
):
    if args.model == "bigram":
        return next_char_dataset(text, context_size=args.context_size, vocab=vocab)
    if args.model == "embedding":
        return next_char_index_dataset(
            text,
            context_size=args.context_size,
            vocab=vocab,
        )
    if args.model == "transformer":
        return next_char_sequence_dataset(
            text,
            context_size=args.context_size,
            vocab=vocab,
        )
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
            feed_forward_activation=args.activation,
            tie_embeddings=args.tie_embeddings,
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


def split_train_validation_text(
    text: str,
    args: argparse.Namespace,
) -> tuple[str, str | None]:
    if args.validation_chars < 0:
        raise ValueError("validation_chars must be non-negative")
    if args.validation_chars == 0:
        return text, None
    if args.validation_chars >= len(text):
        raise ValueError("validation_chars must be smaller than text length")

    train_text = text[: -args.validation_chars]
    validation_text = text[-args.validation_chars :]
    if len(train_text) <= args.context_size:
        raise ValueError("training text must be longer than context_size")
    if len(validation_text) <= args.context_size:
        raise ValueError("validation text must be longer than context_size")
    return train_text, validation_text


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
    parser.add_argument(
        "--activation",
        choices=("relu", "gelu"),
        default=DEFAULT_OPTIONS["activation"],
    )
    parser.add_argument(
        "--tie-embeddings",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_OPTIONS["tie_embeddings"],
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        default=DEFAULT_OPTIONS["eval_only"],
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_OPTIONS["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_OPTIONS["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_OPTIONS["lr"])
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument(
        "--optimizer",
        choices=("sgd", "adam"),
        default=DEFAULT_OPTIONS["optimizer"],
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_OPTIONS["weight_decay"],
    )
    parser.add_argument(
        "--weight-decay-min-ndim",
        type=int,
        default=DEFAULT_OPTIONS["weight_decay_min_ndim"],
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=DEFAULT_OPTIONS["report_every"],
    )
    parser.add_argument(
        "--validation-chars",
        type=int,
        default=DEFAULT_OPTIONS["validation_chars"],
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=DEFAULT_OPTIONS["max_grad_norm"],
    )
    parser.add_argument("--metrics-file", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-text", default=DEFAULT_OPTIONS["seed_text"])
    parser.add_argument("--seed-file", type=Path, default=DEFAULT_OPTIONS["seed_file"])
    parser.add_argument("--generate", type=int, default=DEFAULT_OPTIONS["generate"])
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_OPTIONS["num_samples"],
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        default=DEFAULT_OPTIONS["generate_only"],
    )
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
    args = apply_preset(parser.parse_args(argv), explicit_options=explicit_options)
    setattr(args, "_explicit_options", explicit_options)
    return args


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
        option_name = name.removeprefix("--").replace("-", "_")
        names.add(option_name)
        if option_name.startswith("no_"):
            names.add(option_name.removeprefix("no_"))
    return names


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
