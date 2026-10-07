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
  projection. The math is identical: stacking every head's query, key, and
  value weights gives nanoGPT's `c_attn` matrix (see
  `nanogpt_state_from_nanograd` in `nanogpt_compare.py`).
- The tied language-model head keeps an output bias; nanoGPT's `lm_head` has
  none.
- Fixed-length contexts: NanoGrad models always see exactly `context_size`
  tokens, while nanoGPT also accepts shorter prompts.
- Xavier-uniform initialization by default instead of nanoGPT's
  `normal(0, 0.02)` with scaled residual projections; `init="gpt2"` selects
  the nanoGPT scheme.
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
  multiple temperature/top-k settings for qualitative sampling sweeps, with an
  optional summary CSV for sorting settings by mean `distinct_2`.
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

## Side-By-Side Runs Against nanoGPT

`nanogpt_compare.py` trains NanoGrad's `CharTransformerModel` and the local
`ref_nanogpt/model.py` `GPT` on the same capped text slice:

- same architecture (the preset's width, heads, and layers; MLP at
  `4 * embedding_dim` because nanoGPT fixes it; tied head with bias)
- the same list of random training batches
- the same AdamW settings, weight decay on matrices only, and global-norm
  clipping written the same way
- one shared top-k/temperature sampler (`sampling.py`) fed the same random
  stream, so equal logits produce equal text

PyTorch is only needed for this script:

```bash
python3 -m venv .venv && .venv/bin/pip install torch
.venv/bin/python nanogpt_compare.py --text-file data/tinyshakespeare/input.txt \
    --max-chars 20000 --validation-chars 2000 --steps 1500 --batch-size 16 \
    --lr 0.01 --eval-every 250 --init shared --output-dir /tmp/nanogpt-compare-shared
```

`--output-dir` writes `metrics.csv` (loss curves), `samples.csv` (top-k
samples from both frameworks), `summary.csv`, and `manifest.json`.

### Shared initialization: same weights in, same text out

With `--init shared`, nanoGPT starts from NanoGrad's exact initial weights
and both run in float64. small-gpt, 20,000 Tiny Shakespeare characters (last
2,000 held out), 1,500 steps of batch 16, lr 0.01, betas (0.9, 0.99):

| step | NanoGrad val loss | nanoGPT val loss |
| ---: | ---: | ---: |
| 250 | 2.7210 | 2.7210 |
| 500 | 2.4960 | 2.4960 |
| 1000 | 2.3924 | 2.3924 |
| 1500 | 2.3305 | 2.3305 |

- Largest validation-loss gap over the whole run: `1.2e-8` (float64
  rounding, compounded over 1,500 Adam steps; it is `4e-16` after 100
  steps).
- Top-k samples (k=5, temperature 0.8) are identical character for character
  in all three samples:

  ```text
  First Cizers ouls ours wour haler ast as the he ater thenore ald hally allll
  walre at the weanll ho hare that the thou she at th
  ```

This makes the comparison an end-to-end correctness check: NanoGrad's
embeddings, attention, LayerNorm, GELU, cross-entropy, backward pass, AdamW,
weight decay, and clipping all compute the same function as PyTorch's.
`test_nanogpt_compare.py` repeats a short version whenever torch is
installed.

### Independent initialization: the fair race

Same run with `--init independent` (nanoGPT uses its own init under
`torch.manual_seed(0)`):

| framework | val loss | val ppl | sample dist-2 | tokens/s (CPU, 1 thread) |
| --- | ---: | ---: | ---: | ---: |
| NanoGrad (pure Python) | 2.3305 | 10.28 | 0.430 | ~1,000 |
| nanoGPT (PyTorch, float64) | 2.2980 | 9.95 | 0.425 | ~17,700 |

```text
NanoGrad: First Cizers angtous anll wir are the then oun walll won thoud han
          he to horl at the ast hear had tord helald hor win tor weln t
nanoGPT:  First Cint at ton\nAs arll wir he and he alle werand his he one at
          are to hong hour he of hoou ale to whatond ath hund to to to h
```

Both learn common short words ("the", "he", "his", "that", "thou") and
word-length structure at this size.

nanoGPT ends 0.03 nats lower on seed 0, but one seed is not enough to
judge. Repeating the run over three seeds (`--seed`, which changes the
batches and both initializations) with both NanoGrad initializers
(`--nanograd-init xavier|gpt2`, where `gpt2` is nanoGPT's `normal(0, 0.02)`
with scaled residual projections):

| seed | NanoGrad xavier | NanoGrad gpt2 | nanoGPT |
| ---: | ---: | ---: | ---: |
| 0 | 2.3305 | 2.3163 | 2.2980 |
| 1 | 2.2677 | 2.2756 | 2.3080 |
| 2 | 2.2936 | 2.2820 | 2.3394 |
| mean | 2.2973 | 2.2913 | 2.3151 |

Seed-to-seed spread (about 0.03 nats) is larger than any gap between the
frameworks. With identical math, as the shared run proves, this is what we
should expect. The GPT-2 initializer is a small, inconsistent improvement
here, so `CharTransformerModel` keeps Xavier as its default and offers
`init="gpt2"` as an option.

## Next Gaps

- Allow prompts shorter than `context_size` (nanoGPT-style cropping) so
  generation can start from a single character.
- Keep tracking train/validation curves on larger capped Tiny Shakespeare
  slices now that the tensor core is roughly 20x faster.
