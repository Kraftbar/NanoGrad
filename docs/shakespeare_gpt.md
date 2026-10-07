# A Baby GPT On Tiny Shakespeare

NanoGrad trains nanoGPT's CPU "baby GPT" recipe on the full Tiny Shakespeare
text and lands next to nanoGPT itself.

```bash
python3 gpt_train.py --text-file data/tinyshakespeare/input.txt \
    --preset baby --workers 4 --out-dir runs/baby
python3 char_demo.py --load-model runs/baby/model.json --generate-only \
    --sample-mode sample --top-k 20 --temperature 0.8 --seed-text "ROMEO:" --generate 300
```

## Setup

The config comes from nanoGPT's README, "I only have a MacBook":

| | |
| --- | --- |
| model | 4 layers, 4 heads, 128 wide, context 64, tied embeddings, GELU, 809,921 parameters |
| data | Tiny Shakespeare, 65-character vocabulary, first 90% train / last 10% validation |
| training | 2000 steps of 12 random windows, AdamW (betas 0.9/0.99, weight decay 0.1 on matrices), clip 1.0 |
| learning rate | warmup to 1e-3 over 100 steps, cosine decay to 1e-4 at step 2000 |
| evaluation | every 250 steps, mean loss over 20 random batches per split |

nanoGPT ran with its own command (`--device=cpu --compile=False
--eval_iters=20 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4
--n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0`). Both
ran on the same 4-core machine.

## Loss Curves

| step | NanoGrad train | NanoGrad val | nanoGPT train | nanoGPT val |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 4.221 | 4.214 | 4.168 | 4.165 |
| 250 | 2.436 | 2.422 | 2.429 | 2.445 |
| 500 | 2.251 | 2.284 | 2.273 | 2.314 |
| 750 | 2.119 | 2.128 | 2.134 | 2.191 |
| 1000 | 2.010 | 2.044 | 1.971 | 2.053 |
| 1250 | 1.914 | 2.009 | 1.876 | 2.009 |
| 1500 | 1.853 | 1.955 | 1.856 | 1.923 |
| 1750 | 1.798 | 1.945 | 1.779 | 1.892 |
| 2000 | 1.759 | **1.921** | 1.765 | **1.886** |

The curves track each other within the noise of a 20-batch estimate. The
two runs use different random initializations and batches, and slightly
different initializers, so they are not expected to match exactly. (With
shared weights and batches they do match: see
[language_nanogpt_comparison.md](./language_nanogpt_comparison.md).)

## Samples

Top-k 20, temperature 0.8, from a newline:

```text
As for of me are comby, not be britters brooken wind.

DUKE VINCENTIO:
I lord in to plaiors, theirs are the chee
How frear the have do not weil were her bother
To shall die with measher in the grace her is wise.
Or heiven me to sowfers for thou dield tiserias:
But pray the to thing hather.
```

```text
DUKE OF ICIORGAUTINUS:
True, and so shee.

OSicont you deh, our cuit: the love besorn them: your the
Why,
```

Speaker names in capitals, verse line lengths, real short words and
Shakespeare-shaped nonsense: the same quality nanoGPT's README shows for
this config ("it's still good fun").

## Speed

| | wall time for 2000 steps |
| --- | --- |
| nanoGPT, PyTorch, 4 CPU threads | 1 min 53 s |
| NanoGrad, 4 worker processes, C kernels, float32 | 3 min 24 s |
| NanoGrad, 4 worker processes, C kernels, float64 | 5 min 42 s |
| NanoGrad, earlier C kernels (plain loops), float64 | 11 min 20 s |
| NanoGrad, first run (C matmul only) | about 2 hours |
| NanoGrad, 1 process, pure Python (estimated from a 46 s step) | about 26 hours |

Every float64 run reproduces the 2-hour run exactly: same loss at every
evaluation, same samples. Only the speed changed. The float32 run ends at
the same validation loss to four decimals (1.9210).

Getting here took:

- the tensor-core rewrite (index maps, im2col, slice reductions)
- C kernels for the inner loops (`kernels.c`, optional; the pure-Python
  versions in `kernels_reference.py` stay the readable reference)
- data-parallel workers for training and evaluation (`parallel.py`)
- freeing each step's graph after `backward()`, and not creating
  backward closures under `no_grad`. Before that fix, workers grew to
  3.5 GB and the first run was killed by the out-of-memory killer.
- storing tensor values as C arrays, so data goes from kernel to kernel
  without copies, plus kernels for elementwise math and a fused Adam
  update ([benchmarks.md](./benchmarks.md))
- a blocked, register-tiled matrix product (`gemm` in kernels.c: packed
  panels, 8 x 32 register tiles, cache blocking), at 40-47 GFLOP/s in
  float64 on one core, about what this CPU's vector units allow, and
  computing all attention heads in one batched product

What remains between NanoGrad and PyTorch here is mostly Python: each op
is a Python call that sets up a kernel and records a backward step.
