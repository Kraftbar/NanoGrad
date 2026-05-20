# NanoGPT Comparison

This note compares NanoGrad's tiny character transformer path with the local
`ref_nanogpt/model.py` reference.

## Current Mapping

| nanoGPT concept | NanoGrad concept |
| --- | --- |
| `wte` token embedding + `wpe` position embedding | `TokenPositionEmbedding` |
| `CausalSelfAttention` | `CausalSelfAttention` |
| pre-norm `Block` with attention and MLP residuals | `TransformerBlock` |
| final `ln_f` | `CharTransformerModel.norm` |
| `lm_head` | `CharTransformerModel.projection` |
| autoregressive `generate` loop | `generate_text` in `char_demo.py` |

## Intentional Simplifications

- Separate per-head Q/K/V projections instead of nanoGPT's fused multi-head
  projection.
- No dropout, optimizer scheduling, mixed precision, or GPU path.
- Character-level vocabulary only.

## Behavior Checks

- Causal tests verify future tokens do not change earlier attention or block
  outputs.
- Transformer training uses per-position logits and shifted sequence targets,
  matching nanoGPT's next-token objective more closely than the earlier
  context-only classifier path.
- Causal attention supports one or more heads through `num_heads`.
- Transformer feed-forward layers can use ReLU or GELU through `--activation`.
- The language-model head can share token embedding weights with
  `--tie-embeddings`.
- The tiny transformer preset runs a capped character demo quickly:
  `python3 char_demo.py --preset tiny-transformer`.
- A more GPT-like smoke preset enables GELU, Adam, weight decay, grad clipping,
  and tied embeddings: `python3 char_demo.py --preset tiny-gpt`.
- `--preset lite-gpt` is the default comparison transformer rung: longer
  context than `tiny-gpt` while staying cheap enough for quick iteration.
- `--preset small-gpt` is the next larger character-level transformer rung:
  wider embeddings, two heads, and two blocks.
- Sampled generation supports temperature, optional top-k filtering, and
  multiple samples per prompt with `--num-samples`, matching the shape of the
  local nanoGPT sampling path.
- Prompts can be provided inline with `--seed-text` or from a local file with
  `--seed-file`, mirroring nanoGPT's file-backed prompt workflow.
- Character runs can report per-epoch train/validation metrics with
  `--report-every` and `--validation-chars`; `--no-shuffle` keeps batch order
  fixed for curve comparisons.
- Reports include token-level perplexity from the cross-entropy loss, and epoch
  metrics can also be written as CSV with `--metrics-file`.
- Tensor training can use SGD or Adam with `--optimizer`, and optionally clip
  global gradient norm or apply weight decay with `--max-grad-norm` and
  `--weight-decay`.
- The GPT-like preset applies weight decay only to matrix-like parameters with
  `--weight-decay-min-ndim 2`, matching nanoGPT's bias/layernorm exclusion in a
  small shape-based way.
- Character checkpoints include vocabulary and architecture metadata needed for
  safe `--save-model` / `--load-model` round trips.
- Saved checkpoints can be evaluated or sampled without another training pass
  by adding `--eval-only`; unless explicitly overridden, the saved model
  settings are restored from the checkpoint.
- Saved checkpoints can also be sampled without supplying the training text:
  `python3 char_demo.py --load-model char.json --generate-only --seed-text Firs`.
- `char_demo.py --samples-file` writes generated checkpoint samples with seed
  prefix, sampling settings, text, and `distinct_2`.
- `char_sample_grid.py` loads a saved checkpoint and writes a CSV grid across
  multiple temperature/top-k settings for qualitative sampling sweeps.
- `char_compare.py` runs the bigram, embedding, and lite GPT configs on the
  same text slice and reports loss, perplexity, validation metrics, speed, and
  one or more generated samples per model. It also reports `dist-2`, a simple
  generated-text distinct-bigram ratio for spotting repetitive samples:
  ```bash
  python3 char_compare.py --text-file data/tinyshakespeare/input.txt --max-chars 128 --validation-chars 32 --output-dir /tmp/nanollm-language-run
  ```
  The samples CSV preserves model metadata, sample index, seed prefix, text,
  and per-sample `distinct_2` so top-k/temperature output can be compared
  outside the console.
  The output directory writes summary, epoch metrics, samples, and a JSON
  manifest for reproducing the run.
- The same preset can run against local Tiny Shakespeare with an explicit seed:
  ```bash
  python3 char_demo.py --preset tiny-transformer --text-file data/tinyshakespeare/input.txt --seed-text Firs
  ```

## Next Gaps

- Compare qualitative top-k sampled output CSVs against nanoGPT after training
  on the same capped text slice.
- Keep tracking train/validation curves on larger capped Tiny Shakespeare slices
  as tensor ops become faster.
