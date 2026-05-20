# Character Language Workflow

This repo keeps the language-model path small enough to inspect by hand while
still supporting the core experiment loop: compare baselines, save a checkpoint,
evaluate it, and sample from it.

See [language_experiment_log.md](./language_experiment_log.md) for the small
local runs that motivate default preset choices.

## Compare Baselines

Run the built-in text smoke comparison:

```bash
python3 char_compare.py
```

Run the same comparison on a capped Tiny Shakespeare slice:

```bash
python3 char_compare.py --text-file data/tinyshakespeare/input.txt --max-chars 128 --validation-chars 32 --output-dir /tmp/nanollm-language-run
```

The comparison reports train/validation loss, perplexity, accuracy, generated
sample `dist-2` diversity, rough throughput with units, and generated samples
per model.
Generated samples start from the training-text prefix unless `--seed-text` or
`--seed-file` is provided.
Add `--num-samples 3` for a broader qualitative check. The default comparison skips the slower
`small-gpt` preset and uses `lite-gpt` as the default transformer rung; include
`small-gpt` explicitly with `--models small-gpt` or in a longer model list. CSV metrics include model type, context size, and parameter
count alongside the epoch metrics. `--samples-file` writes each generated
sample with its model metadata, actual seed prefix, and per-sample `distinct_2`
score for later qualitative comparison. `--summary-file` writes the final table
metrics as one row per model for sorting or plotting. `--output-dir` writes all
three CSVs as `summary.csv`, `metrics.csv`, and `samples.csv`, plus a
`manifest.json` with the text source, generation settings, model configs,
output paths, and final summary rows.

## Train And Save

Train the GPT-like preset and save a checkpoint:

```bash
python3 char_demo.py --preset tiny-gpt --text-file data/tinyshakespeare/input.txt --max-chars 128 --validation-chars 32 --seed-text Firs --save-model /tmp/nanollm-char.json
```

For a slightly larger Tiny Shakespeare run, use `--preset lite-gpt` first, then
`--preset small-gpt` when slower iteration is acceptable. Both keep the same
character-level code path while increasing context length and model width.

The checkpoint stores the vocabulary and model architecture, so later commands
do not need to repeat flags like `--model`, `--context-size`, or
`--embedding-dim` unless intentionally overriding them. Demo output includes
`sample dist-2`, the generated sample distinct-bigram ratio.

## Evaluate

Evaluate a saved checkpoint without another training pass:

```bash
python3 char_demo.py --load-model /tmp/nanollm-char.json --text-file data/tinyshakespeare/input.txt --max-chars 128 --validation-chars 32 --eval-only --seed-text Firs
```

## Generate

Sample directly from a checkpoint without supplying the training text:

```bash
python3 char_demo.py --load-model /tmp/nanollm-char.json --generate-only --seed-text Firs --sample-mode sample --temperature 0.8 --top-k 8 --num-samples 3 --samples-file /tmp/nanollm-char-samples.csv
```

Prompts can also come from a local file:

```bash
python3 char_demo.py --load-model /tmp/nanollm-char.json --generate-only --seed-file /tmp/prompt.txt --generate 80
```

Sweep checkpoint sampling settings into one CSV:

```bash
python3 char_sample_grid.py --load-model /tmp/nanollm-char.json --seed-text Firs --temperatures 0.7 0.8 1.0 --top-k none 4 8 --output-dir /tmp/nanollm-char-grid
```

The grid output directory contains `samples.csv`, `summary.csv`, and
`manifest.json`.

## Benchmark

Time a tiny transformer sequence forward/backward pass:

```bash
python3 language_benchmark.py --repeat 3
```

Time the larger shape with a smaller batch:

```bash
python3 language_benchmark.py --preset lite-gpt --repeat 3
python3 language_benchmark.py --preset small-gpt --batch-size 1 --repeat 3
```
