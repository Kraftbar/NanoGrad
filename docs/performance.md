# Performance Notes

NanoGrad stays pure Python with no dependencies. The speedups below come from
doing less Python work per number, not from hiding the math. Every op still
reads as its equation, and `test_gradcheck.py` verifies each backward rule
against finite differences.

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

Average pooling is written as a sum of shifted images: for each offset
inside the window, add one strided slice per output row. It no longer needs
a table of every window's pixel indices. Building that table dominated the
first call on each new image size: 0.61 s versus 0.12 s for a 5x5 box filter
on a 238x318 image. Once the table was cached, the two were close (0.107 s
versus 0.092 s).

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

## Where The Time Goes Now

Profiling `small-gpt` fwd+bwd shows about two thirds of the time inside
`math.sumprod` and `_dot_table`, which is the arithmetic itself. The next
costs are GELU (`math.erf`), LayerNorm, and elementwise adds. Further big
wins would need a compiled kernel or numpy, which this project deliberately
avoids.
