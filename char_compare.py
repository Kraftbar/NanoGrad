"""Compare tiny character language-model configs on one text split."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from char_demo import (
    DEFAULT_OPTIONS,
    DEFAULT_TEXT,
    PRESETS,
    build_dataset,
    build_model,
    generate_samples,
    generation_seed_text,
    load_text,
    split_train_validation_text,
    text_source,
)
from metrics import perplexity
from text import CharVocab, distinct_ngram_ratio, mean_distinct_ngram_ratio
from train import (
    TrainingSummary,
    train_tensor_multiclass_dataset,
    train_tensor_sequence_multiclass_dataset,
)


COMPARISON_CONFIGS = {
    "bigram": {
        "model": "bigram",
        "context_size": 4,
        "epochs": 1,
        "batch_size": 4,
        "lr": 0.2,
        "optimizer": "sgd",
    },
    "embedding": {
        "model": "embedding",
        "context_size": 4,
        "embedding_dim": 8,
        "epochs": 1,
        "batch_size": 4,
        "lr": 0.05,
        "optimizer": "adam",
    },
    "tiny-gpt": PRESETS["tiny-gpt"],
    "lite-gpt": PRESETS["lite-gpt"],
    "small-gpt": PRESETS["small-gpt"],
}
DEFAULT_MODELS = ["bigram", "embedding", "lite-gpt"]


@dataclass(frozen=True)
class ComparisonResult:
    """Final metrics from one comparable character-model run."""

    name: str
    model: str
    context_size: int
    parameters: int
    train_loss: float
    train_perplexity: float
    accuracy: float
    validation_loss: float | None
    validation_perplexity: float | None
    validation_accuracy: float | None
    elapsed_seconds: float
    examples_per_second: float | None
    sample_distinct_2: float
    generated_samples: list[str]


def run(args: argparse.Namespace) -> None:
    apply_output_dir(args)
    text = load_text(args)
    config_args = [
        comparison_args(args, name)
        for name in args.models
    ]
    split_args = argparse.Namespace(
        validation_chars=args.validation_chars,
        context_size=max(config.context_size for config in config_args),
    )
    train_text, validation_text = split_train_validation_text(text, split_args)
    vocab = CharVocab.from_text(text)

    all_records = []
    results = []
    for name, model_args in zip(args.models, config_args):
        result, records = run_one_comparison(
            name,
            model_args,
            train_text,
            validation_text,
            vocab,
            capture_records=args.metrics_file is not None,
        )
        results.append(result)
        all_records.extend(records)

    print("Character language-model comparison")
    print(f"text source:   {text_source(args)}")
    print(f"text length:   {len(text)}")
    print(f"train length:  {len(train_text)}")
    if validation_text is not None:
        print(f"val length:    {len(validation_text)}")
    print(f"vocab size:    {len(vocab)}")
    print(f"generation:    {args.sample_mode}")
    if args.num_samples != 1:
        print(f"num samples:   {args.num_samples}")
    if args.sample_mode == "sample":
        print(f"temperature:   {args.temperature}")
        if args.top_k is not None:
            print(f"top k:         {args.top_k}")
    if args.output_dir is not None:
        print(f"output dir:    {args.output_dir}")
    print(format_results_table(results))
    for result in results:
        print_result_samples(result)

    if args.summary_file is not None:
        write_summary_records(args.summary_file, summary_records(results))
        print(f"summary file:  {args.summary_file}")
    if args.metrics_file is not None:
        write_epoch_metrics(args.metrics_file, all_records)
        print(f"metrics file:  {args.metrics_file}")
    if args.samples_file is not None:
        write_sample_records(args.samples_file, sample_records(results))
        print(f"samples file:  {args.samples_file}")


def run_one_comparison(
    name: str,
    args: argparse.Namespace,
    train_text: str,
    validation_text: str | None,
    vocab: CharVocab,
    *,
    capture_records: bool,
) -> tuple[ComparisonResult, list[dict]]:
    dataset, _ = build_dataset(train_text, args, vocab=vocab)
    validation_dataset = (
        None
        if validation_text is None
        else build_dataset(validation_text, args, vocab=vocab)[0]
    )
    model = build_model(args, vocab_size=len(vocab))
    train_fn = (
        train_tensor_sequence_multiclass_dataset
        if args.model == "transformer"
        else train_tensor_multiclass_dataset
    )
    records = []
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
        epoch_callback=(
            None
            if not capture_records
            else lambda epoch, epoch_summary: records.append(
                epoch_record(
                    name,
                    args,
                    model.num_parameters(),
                    epoch,
                    epoch_summary,
                )
            )
        ),
    )
    seed_text = usable_seed_text(
        generation_seed_text(args),
        train_text=train_text,
        vocab=vocab,
        context_size=args.context_size,
    )
    sample_args = argparse.Namespace(**vars(args))
    sample_args.seed_text = seed_text
    sample_args.seed_file = None
    generated_samples = generate_samples(model, vocab, sample_args)

    train_loss = summary.evaluation_loss or summary.final_loss
    return (
        ComparisonResult(
            name=name,
            model=args.model,
            context_size=args.context_size,
            parameters=model.num_parameters(),
            train_loss=train_loss,
            train_perplexity=perplexity(train_loss),
            accuracy=summary.accuracy or 0.0,
            validation_loss=summary.validation_loss,
            validation_perplexity=(
                None
                if summary.validation_loss is None
                else perplexity(summary.validation_loss)
            ),
            validation_accuracy=summary.validation_accuracy,
            elapsed_seconds=summary.elapsed_seconds,
            examples_per_second=summary.examples_per_second,
            sample_distinct_2=mean_distinct_ngram_ratio(generated_samples, n=2),
            generated_samples=generated_samples,
        ),
        records,
    )


def comparison_args(args: argparse.Namespace, name: str) -> argparse.Namespace:
    options = {
        **DEFAULT_OPTIONS,
        **COMPARISON_CONFIGS[name],
    }
    for option_name in (
        "epochs",
        "batch_size",
        "lr",
        "optimizer",
        "weight_decay",
        "weight_decay_min_ndim",
        "max_grad_norm",
    ):
        value = getattr(args, option_name)
        if value is not None:
            options[option_name] = value

    options["seed"] = args.seed
    options["no_shuffle"] = args.no_shuffle
    options["generate"] = args.generate
    options["sample_mode"] = args.sample_mode
    options["temperature"] = args.temperature
    options["top_k"] = args.top_k
    options["sample_seed"] = args.sample_seed
    options["num_samples"] = args.num_samples
    options["seed_text"] = args.seed_text
    options["seed_file"] = args.seed_file
    return argparse.Namespace(**options)


def apply_output_dir(args: argparse.Namespace) -> None:
    if args.output_dir is None:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    defaults = {
        "summary_file": "summary.csv",
        "metrics_file": "metrics.csv",
        "samples_file": "samples.csv",
    }
    for option_name, filename in defaults.items():
        if getattr(args, option_name) is None:
            setattr(args, option_name, args.output_dir / filename)


def usable_seed_text(
    seed_text: str | None,
    *,
    train_text: str,
    vocab: CharVocab,
    context_size: int,
) -> str:
    if (
        seed_text is not None
        and len(seed_text) >= context_size
        and all(char in vocab.stoi for char in seed_text[-context_size:])
    ):
        return seed_text
    if len(train_text) < context_size:
        raise ValueError("training text is shorter than context_size")
    return train_text[:context_size]


def epoch_record(
    name: str,
    args: argparse.Namespace,
    parameters: int,
    epoch: int,
    summary: TrainingSummary,
) -> dict:
    train_loss = summary.evaluation_loss or summary.final_loss
    return {
        "model": name,
        "model_type": args.model,
        "context_size": args.context_size,
        "parameters": parameters,
        "epoch": epoch,
        "loss": train_loss,
        "perplexity": perplexity(train_loss),
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
        "model",
        "model_type",
        "context_size",
        "parameters",
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


def summary_records(results: list[ComparisonResult]) -> list[dict]:
    return [
        {
            "model": result.name,
            "model_type": result.model,
            "context_size": result.context_size,
            "parameters": result.parameters,
            "train_loss": result.train_loss,
            "train_perplexity": result.train_perplexity,
            "accuracy": result.accuracy,
            "val_loss": result.validation_loss,
            "val_perplexity": result.validation_perplexity,
            "val_accuracy": result.validation_accuracy,
            "sample_distinct_2": result.sample_distinct_2,
            "elapsed_seconds": result.elapsed_seconds,
            "examples_per_second": result.examples_per_second,
            "sample_count": len(result.generated_samples),
        }
        for result in results
    ]


def write_summary_records(path: str | Path, records: list[dict]) -> None:
    fieldnames = [
        "model",
        "model_type",
        "context_size",
        "parameters",
        "train_loss",
        "train_perplexity",
        "accuracy",
        "val_loss",
        "val_perplexity",
        "val_accuracy",
        "sample_distinct_2",
        "elapsed_seconds",
        "examples_per_second",
        "sample_count",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def sample_records(results: list[ComparisonResult]) -> list[dict]:
    records = []
    for result in results:
        for sample_index, sample in enumerate(result.generated_samples, start=1):
            records.append({
                "model": result.name,
                "model_type": result.model,
                "context_size": result.context_size,
                "parameters": result.parameters,
                "sample_index": sample_index,
                "distinct_2": distinct_ngram_ratio(sample, n=2),
                "sample": sample,
            })
    return records


def write_sample_records(path: str | Path, records: list[dict]) -> None:
    fieldnames = [
        "model",
        "model_type",
        "context_size",
        "parameters",
        "sample_index",
        "distinct_2",
        "sample",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def print_result_samples(result: ComparisonResult) -> None:
    if len(result.generated_samples) == 1:
        print(f"{result.name} generated: {result.generated_samples[0]!r}")
        return
    for index, sample in enumerate(result.generated_samples, start=1):
        print(f"{result.name} generated {index}: {sample!r}")


def format_results_table(results: list[ComparisonResult]) -> str:
    headers = [
        "name",
        "ctx",
        "params",
        "train loss",
        "train ppl",
        "val loss",
        "val ppl",
        "acc",
        "val acc",
        "dist-2",
        "items/s",
    ]
    rows = [
        [
            result.name,
            str(result.context_size),
            str(result.parameters),
            f"{result.train_loss:.4f}",
            f"{result.train_perplexity:.2f}",
            _optional_float(result.validation_loss),
            _optional_float(result.validation_perplexity, digits=2),
            f"{result.accuracy:.3f}",
            _optional_float(result.validation_accuracy, digits=3),
            f"{result.sample_distinct_2:.3f}",
            _optional_float(result.examples_per_second, digits=1),
        ]
        for result in results
    ]
    widths = [
        max(len(row[index]) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    lines = [
        "  ".join(
            value.ljust(width)
            for value, width in zip(headers, widths)
        ),
        "  ".join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append(
            "  ".join(
                value.ljust(width)
                for value, width in zip(row, widths)
            )
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(COMPARISON_CONFIGS),
        default=DEFAULT_MODELS,
    )
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--max-chars", type=int, default=128)
    parser.add_argument("--validation-chars", type=int, default=9)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--optimizer", choices=("sgd", "adam"))
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--weight-decay-min-ndim", type=int)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-text")
    parser.add_argument("--seed-file", type=Path)
    parser.add_argument("--generate", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--sample-mode", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--summary-file", type=Path)
    parser.add_argument("--metrics-file", type=Path)
    parser.add_argument("--samples-file", type=Path)
    return parser.parse_args(argv)


def _optional_float(value: float | None, *, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
