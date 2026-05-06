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

- Single attention head instead of fused multi-head QKV projection.
- ReLU feed-forward activation instead of GELU.
- No dropout, optimizer scheduling, mixed precision, or GPU path.
- Character-level vocabulary only.
- The current demo predicts one next character from a full context, while
  nanoGPT trains logits at every sequence position.
- The output projection is not weight-tied to token embeddings.

## Behavior Checks

- Causal tests verify future tokens do not change earlier attention or block
  outputs.
- The tiny transformer preset runs a capped character demo quickly:
  `python3 char_demo.py --preset tiny-transformer`.
- The same preset can run against local Tiny Shakespeare with an explicit seed:
  ```bash
  python3 char_demo.py --preset tiny-transformer --text-file data/tinyshakespeare/input.txt --seed-text Firs
  ```

## Next Gaps

- Add per-position logits and loss for a closer GPT training objective.
- Add multi-head attention once the single-head path is stable.
- Add top-k sampling and better generation controls.
- Compare training curves on the same capped Tiny Shakespeare slice.
