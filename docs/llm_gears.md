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
| bidirectional attention | `causal=False` | for encoders: ViT (`vit.py`) and BERT (`bert.py`) |
| masked language modeling | `bert.py` | BERT's training task |

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
