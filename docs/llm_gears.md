# LLM Gears

The character GPT in `language.py` follows GPT-2 / nanoGPT. Real language
models add a few more parts. Each one is here in readable form, tested, and
checked against PyTorch where PyTorch has an equivalent.

| part | where | what it does |
| --- | --- | --- |
| byte-pair encoding | `bpe.py` | tokens for frequent character sequences instead of single characters |
| KV cache | `language.py` (`step_logits`) | generation adds one row of work per token instead of re-running the context |
| RMSNorm | `tensor_nn.rms_norm` | LLaMA's simpler LayerNorm: no mean, no bias |
| rotary position embeddings | `tensor.rotary_embedding` | positions as rotations of queries and keys |
| SwiGLU | `language.py` (feed-forward) | gated feed-forward: `silu(x W_in) * (x W_up)` |
| grouped-query attention | `num_kv_heads` / `--kv-heads` | query heads share key/value heads: a smaller KV cache (LLaMA 3, Mistral) |
| bidirectional attention | `causal=False` | for encoders: ViT (`vit.py`) and BERT (`bert.py`) |
| masked language modeling | `bert.py` | BERT's training task |

## Results: the Baby GPT on Tiny Shakespeare

Every row is the same recipe as nanoGPT's baby GPT (about 0.8M parameters,
2,000 steps of 12 x 64 tokens, AdamW, warmup + cosine; 4 worker processes
on a 4-core CPU), changing one thing at a time. Bits per character on the
validation text make character and BPE models comparable (lower is better).

| model | val bits / char | val loss (nats / token) | time |
| --- | ---: | ---: | ---: |
| GPT-2 parts, characters (`--preset baby`) | 2.77 | 1.921 | 11 min |
| nanoGPT itself, PyTorch, same recipe | 2.72 | 1.886 | 2 min |
| GPT-2 parts, BPE 512 tokens (`--tokenizer bpe`) | 2.44 | 3.180 | 8 min |
| LLaMA parts, characters (`--preset baby-llama`) | 2.44 | 1.693 | 7 min |
| 2-layer LSTM, characters (`--preset baby-lstm`, 875K parameters) | 2.71 | 1.875 | 24 min |

The LSTM (Karpathy's char-rnn at the same size and recipe) ends level
with the GPT-2-style transformer, but takes three times as long: its 64
steps per window run one after another, while attention handles all 64
positions in a few large matrix products.

Two independent improvements of about the same size:

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
