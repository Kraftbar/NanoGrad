# Performance Notes

NanoGrad needs nothing but Python. This page covers the first round of
speedups, to the pure-Python core: doing less Python work per number, not
hiding the math. Every op still reads as its equation, and
`test_gradcheck.py` verifies each backward rule against finite differences.

The second round added optional C kernels (`kernels.c`) with the pure-Python
loops kept beside them as the readable reference (`kernels_reference.py`),
and tensors that store their values as C arrays. That round, and how
NanoGrad now compares with PyTorch, is in [benchmarks.md](./benchmarks.md).
Some code snippets below show the pure-Python versions from the first round.

## Before / After

Same machine, same commands, original `main` versus this branch. Lower is
better.

| Benchmark | Before | After | Speedup |
| --- | --- | --- | --- |
| `benchmark.py` matmul 32x64 @ 64x32, fwd+bwd | 32.1 ms | 3.8 ms | 8.4x |
| `benchmark.py` conv2d 4x1x12x12, 6 filters, fwd+bwd | 8.6 ms | 3.3 ms | 2.6x |
| `language_benchmark.py` tiny-gpt, batch 4 | 52 ms | 3.1 ms | 17x |
| `language_benchmark.py --preset lite-gpt --batch-size 16` | 700 ms | 26 ms | 27x |
| `language_benchmark.py --preset small-gpt --batch-size 16` | 1946 ms | 89 ms | 22x |
| `char_demo.py --preset small-gpt --epochs 2` throughput | 60 tok/s | 1215 tok/s | 20x |
| `mnist_smoke_demo.py` throughput | 2264 samples/s | 7278 samples/s | 3.2x |

The `char_demo` run prints the same losses (to 6 decimals) and the same
generated text before and after.

For scale: on `small-gpt`, single-threaded PyTorch on CPU (float64) runs
about 16x faster than NanoGrad (see `nanogpt_compare.py`). The remaining gap
is the cost of one Python-level operation per multiply-add.

## What Changed, And Why It Is Fast

### 1. Broadcasting is an index map, built once

The old code turned every output index into coordinates and back, once per
element per operand, inside the loop. Now each (output shape, input shape)
pair gets one cached flat "index map". A broadcast axis is just stride 0:

```python
# tensor.py
@lru_cache(maxsize=256)
def _broadcast_index_map(out_shape, source_shape):
    strides = ... 0 for broadcast axes, row-major stride otherwise ...
    return _strided_indices(out_shape, strides)

left_values = _expand(left, out_shape)      # no copy if shapes already match
right_values = _expand(right, out_shape)
data = [a + b for a, b in zip(left_values, right_values)]
```

The backward pass sums gradients through the same map
(`_reduce_to_shape`). Permutes use the same idea (`_permute_index_map`).

### 2. Every matrix product is one dot-product table

(Today every matrix product is one call to `kernels.gemm`. Its Python
version in `kernels_reference.py` still works this way: each output entry is
one dot product of a row slice and a column slice.)

```python
def _dot_table(row_vectors, column_vectors):
    out = []
    for row in row_vectors:
        out.extend([_dot(row, column) for column in column_vectors])
    return out
```

`_dot` is `math.sumprod` (C speed, Python 3.12+), falling back to
`sum(map(mul, a, b))`. The forward and both backward products are the same
call with different row and column views:

```text
C  = A  @ B    ->  _dot_table(rows(A),     columns(B))
dA = dC @ B^T  ->  _dot_table(rows(dC),    rows(B))
dB = A^T @ dC  ->  _dot_table(columns(A),  columns(dC))
```

`matmul` follows numpy's `@` rules, including batches
`(..., n, k) @ (..., k, m)`. When the right side is a plain matrix, all
leading axes fold into rows, so `linear` on `(batch, time, features)` is a
single table.

### 3. Convolution is im2col

Gather every kernel-sized image patch into a row (index lists cached per
shape), then `out = dot_table(kernels, patches)`. The backward pass is two
more tables plus a scatter-add of patch gradients back onto pixels.

Pooling reads every window through a cached table of pixel offsets, like
im2col (`kernels.window_mean` and `window_max`).

### 4. Reductions and softmax work on lanes

A "lane" is every value that differs only along the reduced axis. Each lane
is a Python `slice(start, stop, step)`, so `data[lane]` reads it at C speed
and `grad[lane] = [...]` writes it back.

### 5. The attention shape bug was the biggest single win

Attention used to flatten the batch and score every token against every
token *in the whole batch*: a `(B*T, B*T)` matrix with a mask to hide other
sequences. Now it is batched per sequence, `(B, T, T)`, just like nanoGPT.
This is `B` times less work, and the mask is a `(T, T)` constant broadcast
over the batch.

### 6. Fewer, fatter ops in the language model

- Embedding lookup is `gather_rows` (copy rows, scatter-add gradients)
  instead of a one-hot matrix times the table.
- LayerNorm is one fused op with its backward written out in the docstring,
  instead of about ten broadcasted ops. A test checks it against the composed
  version.
- Optimizers update a whole parameter list per comprehension.
- Evaluation and sampling run under `no_grad()`, which skips recording the
  graph.

### 7. Smaller things

- `_op_result` builds op outputs without re-validating or copying data.
- Intermediate gradients start as `None` and are created on first use
  instead of being zero-filled and then added to.
- The backward pass sorts the graph iteratively, so deep graphs no longer
  hit Python's recursion limit.

## Using Every Core: Data Parallelism

Python runs one thread of bytecode at a time, so `parallel.py` uses worker
processes instead. Each worker holds a model copy; every step the main
process sends the current weights and a slice of the batch to each worker,
and averages the returned gradients weighted by each slice's target count.
That is exactly the whole-batch gradient (tests compare it to a single
process to 1e-12), so training runs are the same, only faster:

| Run | 1 process | 4 workers | Speedup |
| --- | --- | --- | --- |
| one fwd+bwd, 88k-param transformer, batch 32 x 32 tokens | 6.9 s | 2.1 s | 3.3x |
| `char_demo.py --preset small-gpt --batch-size 32`, 2 epochs | 1152 tok/s | 2809 tok/s | 2.4x |

The small preset gains less because copying weights and gradients is a
bigger share of a cheap step. Use `--workers 4` with `char_demo.py`.

## Threads Inside The Kernels

The C kernels can also use threads themselves (OpenMP; `kernels_c.py`
builds with `-fopenmp` when the compiler supports it).
`NANOGRAD_THREADS=8` or `kernels.set_threads(8)` turns them on; the
default is one thread. The matrix products, the four convolution kernels
and the row softmax use them, and the rule that keeps results exact is
the same in each: **threads split the output, never a sum**.

- A matrix product is cut into bands of whole register tiles: column
  tiles, or row tiles when there are more of those, or whole batch
  entries when each has its own output. A thread runs the same tile code
  on its band as one thread would on the whole matrix, so each output
  number is summed in the same order. When batch entries are summed into
  one output (a weight gradient), a thread adds every entry into its band,
  in order.
- The kernel gradient of a convolution sums over images, so it is split
  the same way: blocks of the gradient, each thread adding every image in
  order. The image gradient is split by image only: neighbouring patches
  overlap, so splitting an image would let two threads add to one pixel.
- The softmax shares out whole rows.

So the result is the same, bit for bit, for any thread count, and the
float64 PyTorch comparisons hold with threads on. Blocks are handed out
four per thread as threads come free (`schedule(dynamic)`), so a thread
slowed by another program on its core does fewer of them. A job is only
shared when each thread gets about a million multiply-adds (25
microseconds of work); below that, starting the threads costs more than
they save. That sets how many blocks a job is cut into, but a parallel
region always asks for every thread: libgomp ends surplus threads when a
region asks for fewer than the last one, and starts new ones for the next
big job. Before this, every thread it replaced leaked its packing buffers
(about 0.3 MB each), which killed a 150-epoch CIFAR-10 run at 19 GB.
Each thread keeps its packing buffers between calls, and a thread that
ends frees them.

Threads and `parallel.py`'s worker processes do not mix: a forked child
cannot use the parent's OpenMP threads (libgomp's pool does not survive
`fork`). `kernels_c.py` therefore sets a forked child to one thread
(`os.register_at_fork`), and with one thread the kernels never start an
OpenMP team. Use one or the other.

## A Whole LSTM Layer (And The CRF) In One Op

An LSTM layer used to be built from per-step ops: a matrix product
`h W_h^T`, an add, the fused `lstm_memory` and `lstm_output` kernels, and
a reshape, six Python ops and autograd nodes per time step, each paying
for buffer wrapping and a fresh zero array. `rnn.lstm_sequence` runs the
whole layer as one op: its C kernels walk the steps forward, then
backpropagate through time. They make the same kernel calls in the same
order (the same gemms into zeroed buffers, the same two-term sums, `dW`
summed latest step first), so results are identical bit for bit in
float64 and float32 (`test_rnn.SequenceTests`).

The CRF's forward algorithm got the same treatment: `tagger.crf_forward`
replaces eight Python ops per word (`alpha - m`, exp, a matrix product,
log, two adds and the reshapes) with one op whose C kernels make the same
calls in the same order, so log Z and every gradient are again the same
bit for bit (`test_tagger`, `test_same_bits_as_the_per_word_ops`).

UD English EWT tagger, training step on one core (best of three runs of
60 batches of 32, idle machine, 2026-10-14):

| tagger | per-step ops | fused LSTM | fused CRF | both | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| BiLSTM-CRF | 53.7 ms | 44.3 ms | 49.4 ms | 40.4 ms | 1.33x |
| + character BiLSTM | 115.7 ms | 79.4 ms | 110.5 ms | 75.5 ms | 1.53x |

The character model gains most: it runs many short LSTMs (one per word
length in a batch), where per-step overhead was most of the cost. What
is left is the tagger's other small ops (embedding gathers, dropout,
Adam).

## Where The Time Goes Now

See [benchmarks.md](./benchmarks.md#how-the-gap-closed). In short, with the
C kernels on, most of a GPT step is now matrix products. The tiled `gemm`
runs at about 47 GFLOP/s in float64 against PyTorch's 62 on the cloud
Xeon, and at or above PyTorch on the Ryzen 7 3700X (53 vs 51 on one core).
