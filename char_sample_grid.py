"""Generate a temperature/top-k sample grid from a saved character checkpoint."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from char_demo import (
    DEFAULT_OPTIONS,
    apply_checkpoint_config,
    build_model,
    checkpoint_vocab,
    generate_samples,
    generation_seed_text,
    load_checkpoint,
)
from text import distinct_ngram_ratio


def run(args: argparse.Namespace) -> None:
    records = sample_grid_records(args)
    write_sample_grid(args.samples_file, records)
    if args.summary_file is not None:
        write_grid_summary(args.summary_file, grid_summary_records(records))

    print("Character sample grid")
    print(f"checkpoint:    {args.load_model}")
    if args.output_dir is not None:
        print(f"output dir:    {args.output_dir}")
    print(f"samples file:  {args.samples_file}")
    if args.summary_file is not None:
        print(f"summary file:  {args.summary_file}")
    print(f"temperatures:  {', '.join(str(value) for value in args.temperatures)}")
    print(f"top k:         {', '.join(top_k_label(value) for value in args.top_k)}")
    print(f"rows:          {len(records)}")


def sample_grid_records(args: argparse.Namespace) -> list[dict]:
    vocab = checkpoint_vocab(args.load_model)
    model_args = checkpoint_model_args(args)
    model = build_model(model_args, vocab_size=len(vocab))
    load_checkpoint(
        args.load_model,
        model,
        vocab,
        model_name=model_args.model,
    )

    records = []
    for temperature in args.temperatures:
        for top_k in args.top_k:
            sample_args = argparse.Namespace(**vars(model_args))
            sample_args.seed_text = args.seed_text
            sample_args.seed_file = args.seed_file
            sample_args.generate = args.generate
            sample_args.num_samples = args.num_samples
            sample_args.sample_mode = args.sample_mode
            sample_args.temperature = temperature
            sample_args.top_k = top_k
            sample_args.sample_seed = args.sample_seed

            seed_text = generation_seed_text(sample_args)
            samples = generate_samples(model, vocab, sample_args)
            for sample_index, sample in enumerate(samples, start=1):
                records.append({
                    "checkpoint": args.load_model,
                    "model_type": model_args.model,
                    "context_size": model_args.context_size,
                    "parameters": model.num_parameters(),
                    "temperature": temperature,
                    "top_k": top_k,
                    "sample_index": sample_index,
                    "seed_text": seed_text,
                    "sample_mode": args.sample_mode,
                    "sample_seed": args.sample_seed,
                    "distinct_2": distinct_ngram_ratio(sample, n=2),
                    "sample": sample,
                })
    return records


def checkpoint_model_args(args: argparse.Namespace) -> argparse.Namespace:
    model_args = argparse.Namespace(**DEFAULT_OPTIONS)
    model_args.seed = args.seed
    model_args._explicit_options = set()
    apply_checkpoint_config(model_args, args.load_model)
    return model_args


def write_sample_grid(path: str | Path, records: list[dict]) -> None:
    fieldnames = [
        "checkpoint",
        "model_type",
        "context_size",
        "parameters",
        "temperature",
        "top_k",
        "sample_index",
        "seed_text",
        "sample_mode",
        "sample_seed",
        "distinct_2",
        "sample",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def grid_summary_records(records: list[dict]) -> list[dict]:
    groups = {}
    for record in records:
        key = (
            record["checkpoint"],
            record["model_type"],
            record["context_size"],
            record["parameters"],
            record["temperature"],
            record["top_k"],
            record["sample_mode"],
            record["sample_seed"],
        )
        groups.setdefault(key, []).append(record)

    summaries = []
    for (
        checkpoint,
        model_type,
        context_size,
        parameters,
        temperature,
        top_k,
        sample_mode,
        sample_seed,
    ), group in groups.items():
        summaries.append({
            "checkpoint": checkpoint,
            "model_type": model_type,
            "context_size": context_size,
            "parameters": parameters,
            "temperature": temperature,
            "top_k": top_k,
            "sample_mode": sample_mode,
            "sample_seed": sample_seed,
            "sample_count": len(group),
            "mean_distinct_2": sum(
                record["distinct_2"]
                for record in group
            ) / len(group),
        })
    return summaries


def write_grid_summary(path: str | Path, records: list[dict]) -> None:
    fieldnames = [
        "checkpoint",
        "model_type",
        "context_size",
        "parameters",
        "temperature",
        "top_k",
        "sample_mode",
        "sample_seed",
        "sample_count",
        "mean_distinct_2",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def apply_output_dir(args: argparse.Namespace) -> None:
    if args.output_dir is None:
        return
    if args.samples_file is None:
        args.samples_file = args.output_dir / "samples.csv"
    if args.summary_file is None:
        args.summary_file = args.output_dir / "summary.csv"


def parse_top_k(value: str) -> int | None:
    if value.lower() in ("none", "null", "-"):
        return None
    try:
        top_k = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("top_k must be an integer or 'none'") from error
    if top_k <= 0:
        raise argparse.ArgumentTypeError("top_k must be positive")
    return top_k


def parse_temperature(value: str) -> float:
    try:
        temperature = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("temperature must be a number") from error
    if temperature <= 0.0:
        raise argparse.ArgumentTypeError("temperature must be positive")
    return temperature


def top_k_label(value: int | None) -> str:
    return "none" if value is None else str(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--samples-file", type=Path)
    parser.add_argument("--summary-file", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-text", default=DEFAULT_OPTIONS["seed_text"])
    parser.add_argument("--seed-file", type=Path)
    parser.add_argument("--generate", type=int, default=80)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--sample-mode", choices=("sample", "greedy"), default="sample")
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=parse_temperature,
        default=[0.7, 0.8, 1.0],
    )
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=parse_top_k,
        default=[None, 4, 8],
    )
    args = parser.parse_args(argv)
    apply_output_dir(args)
    if args.samples_file is None:
        parser.error("--samples-file is required unless --output-dir is provided")
    return args


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
