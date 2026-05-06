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
- ReLU feed-forward activation instead of GELU.
- No dropout, optimizer scheduling, mixed precision, or GPU path.
- Character-level vocabulary only.
- The output projection is not weight-tied to token embeddings.

## Behavior Checks

- Causal tests verify future tokens do not change earlier attention or block
  outputs.
- Transformer training uses per-position logits and shifted sequence targets,
  matching nanoGPT's next-token objective more closely than the earlier
  context-only classifier path.
- Causal attention supports one or more heads through `num_heads`.
- The tiny transformer preset runs a capped character demo quickly:
  `python3 char_demo.py --preset tiny-transformer`.
- Sampled generation supports temperature and optional top-k filtering.
- The same preset can run against local Tiny Shakespeare with an explicit seed:
  ```bash
  python3 char_demo.py --preset tiny-transformer --text-file data/tinyshakespeare/input.txt --seed-text Firs
  ```

## Next Gaps

- Compare top-k sampled output against the nanoGPT sampling path.
- Compare training curves on the same capped Tiny Shakespeare slice.
