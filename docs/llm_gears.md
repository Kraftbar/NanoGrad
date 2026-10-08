# LLM Gears

The character GPT in `language.py` follows GPT-2 / nanoGPT. Real language
models add a few more parts. Each one is here in readable form, tested, and
checked against PyTorch where PyTorch has an equivalent.

| stage | part | where | what it does |
| --- | --- | --- | --- |
| tokens | byte-pair encoding | `bpe.py` | tokens for frequent character sequences instead of single characters |
| architecture | RMSNorm | `tensor_nn.rms_norm` | LLaMA's simpler LayerNorm: no mean, no bias |
| | rotary position embeddings | `tensor.rotary_embedding` | positions as rotations of queries and keys |
| | SwiGLU | `language.py` (feed-forward) | gated feed-forward: `silu(x W_in) * (x W_up)` |
| | grouped-query attention | `num_kv_heads` / `--kv-heads` | query heads share key/value heads: a smaller KV cache (LLaMA 3, Mistral) |
| | sliding-window attention | `attention_window` / `--window` | each token attends to the last W tokens; the KV cache is a rolling buffer (Mistral) |
| | mixture of experts | `moe.py`, `--preset baby-moe` | each token uses 2 of 4 feed-forward experts, chosen by a router (Mixtral) |
| | bidirectional attention | `causal=False` | for encoders: ViT (`vit.py`) and BERT (`bert.py`) |
| training | masked language modeling | `bert.py` | BERT's pretraining task |
| | gradient checkpointing | `recompute.py`, `--recompute` | keep only each block's input; recompute its activations in backward |
| | LoRA | `lora.py`, `bert_finetune.py --lora 4` | fine-tune through small low-rank updates to frozen attention weights |
| | preference tuning (DPO) | `dpo.py` | learn from "A is better than B" pairs directly, no reward model or RL (Rafailov et al.) |
| | distillation | `distill.py` | train a small model on a big one's temperature-softened predictions (Hinton et al.) |
| inference | KV cache | `language.py` (`step_logits`) | generation adds one row of work per token instead of re-running the context |
| | speculative decoding | `speculative.py` | a small model drafts k tokens, the large one checks them in one pass; same output distribution |
| | weight quantization | `quantize.py` | int8 / int4 weights with one scale per 32 numbers: 4-6x smaller |

## Results: the Baby GPT on Tiny Shakespeare

Every row is the same recipe as nanoGPT's baby GPT (about 0.8M parameters,
2,000 steps of 12 x 64 tokens, AdamW, warmup + cosine; 4 worker processes
on a 4-core CPU), changing one thing at a time. Bits per character on the
validation text make character and BPE models comparable (lower is better).
Times are wall-clock with the kernels of the day; the kernels got faster
between runs, so they are rough.

| model | val bits / char | val loss (nats / token) | time |
| --- | ---: | ---: | ---: |
| GPT-2 parts, characters (`--preset baby`) | 2.77 | 1.921 | 6 min |
| nanoGPT itself, PyTorch, same recipe | 2.72 | 1.886 | 2 min |
| GPT-2 parts, BPE 512 tokens (`--tokenizer bpe`) | 2.44 | 3.180 | 8 min |
| LLaMA parts, characters (`--preset baby-llama`) | 2.44 | 1.693 | 7 min |
| LLaMA parts, BPE 512 tokens (`--preset baby-llama --tokenizer bpe`) | **2.22** | 2.885 | 6 min |
| 2-layer LSTM, characters (`--preset baby-lstm`, 875K parameters) | 2.71 | 1.875 | 24 min |

The LSTM (Karpathy's char-rnn at the same size and recipe) ends level
with the GPT-2-style transformer, but takes three times as long: its 64
steps per window run one after another, while attention handles all 64
positions in a few large matrix products.

Two independent improvements of about the same size, and they add up:
both together reach 2.22 bits per character.

- **BPE** makes each token cover 1.88 characters, so each of the 2,000
  steps sees about twice as much text.
- **RMSNorm + rotary embeddings + SwiGLU** learn faster from the same
  characters: at step 1,000 the LLaMA-style model is already below the
  GPT-2-style model's final loss.

Samples, LLaMA parts, top-k 20, temperature 0.8:

```text
No, your compector, and they stay may
Or sking the dead of live to there adoes
Atay the servil of his conclude own of some worth thee
To heart up a crack, in sulse of hold,
And brows say is stones: I'll be our friends,
```

## Byte-Pair Encoding

```bash
python3 gpt_train.py --text-file data/tinyshakespeare/input.txt --preset baby \
    --tokenizer bpe --bpe-vocab-size 512 --workers 4 --out-dir runs/baby_bpe
```

`bpe.py` is GPT-2-style byte-level BPE. It starts from the 256 byte values,
so any text can be encoded with no "unknown" token. Text is first split by a
regex into chunks: words with their leading space, numbers, punctuation,
whitespace. Then the most frequent adjacent pair of tokens is merged into a
new token, again and again. Pairs are counted once per *unique* chunk,
weighted by how often the chunk occurs, which is what makes training fast:
512 tokens learned on the 1 MB training split take about 9 seconds.

The first merges on Tiny Shakespeare are `" t"`, `"he"`, `" a"`, `"ou"`, and
later ones are whole words like `" the"`, `" would"`, `" king"`, `" know"`. A
token covers 1.94 characters on average, so a 64-token context sees about
twice as much text as 64 characters.

Losses per token from different tokenizers cannot be compared directly, so
`gpt_train.py` also logs validation **bits per character**: loss per token /
ln 2 / characters per token.

## KV Cache

Without a cache, generating each token runs the whole context through the
model again. With one, each attention layer keeps the keys and values of
the tokens seen so far; a new token computes only its own query, key and
value, appends the key and value, and attends over the cache.
`CharTransformerModel.step_logits` gives exactly the same logits as the full
forward pass (tested to 1e-12).

Learned absolute positions cannot slide along: every cached key was
computed at its old position. So when the context window is full,
generation restarts the cache from the most recent half window. (With RoPE
the cached keys would stay valid, since only distances between tokens
matter, but positions beyond the training context were never seen in
training, so generation restarts the same way.)

## LLaMA's Parts

```bash
python3 gpt_train.py --text-file data/tinyshakespeare/input.txt --preset baby-llama \
    --workers 4 --out-dir runs/baby_llama
```

- **RMSNorm** drops LayerNorm's mean subtraction and bias:
  `y = x / sqrt(mean(x^2) + eps) * weight`. It has its own C kernel pair,
  and its backward formula is written out in the docstring.
- **Rotary position embeddings (RoPE)** add no position vectors. Instead
  every (even, odd) pair of a query or key is rotated by `position * base^(-2i/D)`,
  so the dot product of a query and a key depends only on how far apart they
  are. The op is written as `x * cos + swap(x) * sin`, and its gradient is
  the opposite rotation.
- **SwiGLU** replaces `gelu(x W_in) W_out` with `(silu(x W_in) * x W_up) W_out`.
  With a hidden width of 8/3 of the embedding, as LLaMA uses, the
  parameter count stays about the same as GPT-2's 4x feed-forward.

`torch_twin.py` runs these with independent formulations: RoPE as LLaMA's
reference code does it (multiplying complex numbers), RMSNorm with
`rsqrt`, SwiGLU with `F.silu`. Outputs agree to about 1e-16 and gradients to
about 1e-14.

The `baby-llama` preset is the `baby` GPT (4 layers, 4 heads, 128 wide,
context 64, same training recipe) with these three parts swapped in:
805,377 parameters against 809,921.

## Encoders: ViT and BERT

The same transformer block with `causal=False` is an encoder. Every token
sees the whole sequence:

- `vit.py` cuts an image into patches and treats them as tokens
  ([benchmarks](./benchmarks.md#model-families))
- `bert.py` hides 15% of the characters (80% become [MASK], 10% a random
  character, 10% stay) and predicts them from both sides. `bert_compare.py`
  trains it in NanoGrad and PyTorch from the same weights: after 500 steps
  both reach the same loss and 22.25% masked-character accuracy on held-out
  text.

## Grouped-Query Attention

During generation the KV cache, not the weights, is what fills memory: one
key row and one value row per token, per head, per layer. Grouped-query
attention (Ainslie et al., 2023) lets a group of query heads share one
key/value head. With `--heads 4 --kv-heads 2` the cache is half the size;
`--kv-heads 1` is multi-query attention.

The batched forward pass needs no copies of the shared keys. The G query
heads of a group are stacked along the time axis, so one group's queries
are a (G * T, D) matrix that multiplies its one (T, D) key matrix, and the
causal mask is the (T, T) mask stacked G times:

```text
query (B, H, T, D)  ->  (B * KV, G * T, D)      H = KV * G
scores = query @ key^T                          (B * KV, G * T, T)
out    = softmax(scores + mask) @ value         (B * KV, G * T, D)  ->  (B, H, T, D)
```

A test checks that this equals ordinary attention whose heads hold copies
of the shared weights, and the PyTorch twin computes each head separately.

## Weight Quantization

`quantize.py` rounds every weight matrix to 8, 4 or 3-bit integers, with
one scale per group of 32 weights (symmetric round-to-nearest, the idea
behind llama.cpp's Q8_0/Q4_0 formats), and measures the validation loss of
the LLaMA-style baby model over the whole validation text:

| weights | size | val bits / char | change |
| --- | ---: | ---: | ---: |
| float32 | 3,120 KB | 2.4085 | |
| int8 | 878 KB | 2.4085 | +0.0000 |
| int4 | 488 KB | 2.4229 | +0.0144 |
| int3 | 391 KB | 2.4801 | +0.0716 |

8 bits cost nothing measurable; 4 bits cost 0.6% at a sixth of the size;
at 3 bits the error starts to show. (Smarter schemes such as GPTQ or AWQ
choose the rounding to keep the layer outputs, not the weights, close.)

## Pretraining, Fine-Tuning, Probing And LoRA

`bert_finetune.py` pretrains the BERT-style encoder (2 layers, 64 wide)
for 4,000 steps of masked-character modeling on Tiny Shakespeare, then
trains it to tell which of 8 characters speaks a 64-character window
(chance: 12.5%), and compares against the same classifier on an untrained
encoder. Three ways to adapt the encoder, 1,500 steps each, accuracy on
held-out speeches:

| how the encoder is adapted | trains | pretrained encoder | untrained encoder |
| --- | ---: | ---: | ---: |
| full fine-tuning | every weight | 17.9% | 18.1% |
| LoRA, rank 4 (`--lora 4`) | 5,640 numbers | 16.2% | 17.0% |
| linear probe, encoder frozen (`--probe`) | the classifier | **22.6%** | 19.2% |

Only the probe shows what pretraining learned: its features make the
speaker 3.4 points easier to tell apart. Training the encoder (fully or
through LoRA) on just 1,085 labelled speeches overfits them: training
loss falls to about 1.1 while held-out accuracy stays near 17%, with or
without pretraining. At this scale the honest result is that pretrained
features help a little and fine-tuning needs more labelled data than this
task has.

## Gradient Checkpointing

Training memory is mostly activations: every intermediate result of the
forward pass waits for backward. `recompute(block, x)` runs a block
without recording it, keeps only its input, and runs it again (recording)
when backward reaches it; dropout masks repeat because the block's random
generators are rewound. A test checks that outputs and gradients are
identical. On the baby GPT at batch 32 (`--recompute`):

| | peak memory (one step) | time per step |
| --- | ---: | ---: |
| plain | 643 MB | 643 ms |
| recompute each block | 199 MB | 946 ms |

A third of the memory for half again the time; the theory says a third
more time (one extra forward pass of each block), and the rest is
Python building each block's graph twice.

## Results: context past the training length (`context.py`)

The LLaMA-style baby GPT was trained on 64-character windows, with
rotary positions. It is read at 128 and 256 characters in two ways.
*Extrapolating* uses the raw positions, so the rotation angles go past
those seen in training. *Interpolating* (Chen et al., 2023) scales
positions by 64 / L, so every angle is a trained one. Each way is also
fine-tuned for 300 steps at the new length. The control is the same
fine-tune at 64. Validation loss is in nats per character, over the
whole window and over its last quarter (the positions with the most
history):

| length | method | loss | last quarter |
| ---: | --- | ---: | ---: |
| 64 | as trained | 1.671 | 1.647 |
| 64 | + 300 steps (control) | 1.706 | 1.683 |
| 128 | extrapolate | 1.748 | 1.951 |
| 128 | interpolate | 2.627 | 2.634 |
| 128 | extrapolate + 300 steps | **1.665** | **1.634** |
| 128 | interpolate + 300 steps | 1.679 | 1.653 |
| 256 | extrapolate | 2.005 | 2.373 |
| 256 | interpolate | 3.234 | 3.213 |
| 256 | extrapolate + 300 steps | **1.639** | 1.646 |
| 256 | interpolate + 300 steps | 1.702 | 1.694 |

The control matters. The same 300 steps do not help at 64 characters
(1.671 -> 1.706), so the gains at 128 and 256 come from the longer
history, not from extra training.

Without fine-tuning, both methods get worse. Extrapolation degrades
gradually, mostly at the late positions, where the angles are new.
Interpolation is much worse everywhere, because it squeezes the
fastest-rotating pairs that encode which character came just before.

After a short fine-tune, both use the longer context. At this scale
extrapolation comes out ahead, unlike in the paper. The likely reason is
that with 32-dimensional heads, the slowest rotary pairs turn by only
about 0.05 radians over 256 positions, so extrapolating creates few
truly unseen angles. LLaMA's long-range pairs, at 2K -> 32K tokens, do.

