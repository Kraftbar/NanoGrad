# Character Language Workflow

This repo keeps the language-model path small enough to inspect by hand while
still supporting the core experiment loop: compare baselines, save a checkpoint,
evaluate it, and sample from it.

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
sample `dist-2` diversity, rough throughput, and generated samples per model.
Add `--num-samples 3` for a broader qualitative check. The default comparison skips the slower
`small-gpt` preset; include it explicitly with `--models small-gpt` or in a
longer model list. CSV metrics include model type, context size, and parameter
count alongside the epoch metrics. `--samples-file` writes each generated
sample with its model metadata and per-sample `distinct_2` score for later
qualitative comparison. `--summary-file` writes the final table metrics as one
row per model for sorting or plotting. `--output-dir` writes all three CSVs as
`summary.csv`, `metrics.csv`, and `samples.csv`.

## Train And Save

Train the GPT-like preset and save a checkpoint:

```bash
python3 char_demo.py --preset tiny-gpt --text-file data/tinyshakespeare/input.txt --max-chars 128 --validation-chars 32 --seed-text Firs --save-model /tmp/nanollm-char.json
```

For a slightly larger Tiny Shakespeare run, use `--preset small-gpt` and pass an
8-character seed such as `--seed-text "First Ci"`. It keeps the same
character-level code path but increases context length, embedding width, heads,
and layers.

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
python3 char_demo.py --load-model /tmp/nanollm-char.json --generate-only --seed-text Firs --sample-mode sample --temperature 0.8 --top-k 8 --num-samples 3
```

Prompts can also come from a local file:

```bash
python3 char_demo.py --load-model /tmp/nanollm-char.json --generate-only --seed-file /tmp/prompt.txt --generate 80
```

## Benchmark

Time a tiny transformer sequence forward/backward pass:

```bash
python3 language_benchmark.py --repeat 3
```

Time the larger shape with a smaller batch:

```bash
python3 language_benchmark.py --preset small-gpt --batch-size 1 --repeat 3
```
